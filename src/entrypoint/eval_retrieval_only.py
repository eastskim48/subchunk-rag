from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer, logging

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compressor.factory import compress_docs, initialize_compressor
from compressor.output import (
    selected_context_segment_count,
    selected_context_text,
)
from evidence_coverage import (
    TextEvidenceCoverageScorer,
    load_text_evidence_labels,
    summarize_text_evidence_records,
)
from utils import parse_json_query
from vectordb import ChromaDB


def actual_token_count(text: str, tokenizer) -> int:
    if not text.strip():
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_jsonl_query_records(
    path: Path, probe_query_limit: int | None = None
) -> list[dict[str, Any]]:
    if probe_query_limit is not None and probe_query_limit <= 0:
        raise ValueError(f"probe_query_limit must be positive, got {probe_query_limit}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if probe_query_limit is not None and line_index >= probe_query_limit:
                break
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} contains a query row that is not an object")
            records.append({"id": payload.get("id"), "query": parse_json_query(line)})
    return records


def docs_full_text(docs) -> str:
    return "\n\n".join(getattr(doc, "text", "") for doc in docs)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_retrieval_only_evaluation(
    *,
    dataset: str,
    query_records: list[dict[str, Any]],
    vectordb,
    tokenizer,
    query_file: str,
    db_dir: str,
    evidence_path: Path,
    top_k: int,
    bsz: int,
    compress_method: str | None,
    model_name: str,
    setup_time: float,
    colbert_query_config: dict[str, Any] | None,
    run_start: float,
    output_file: str,
    details_file: str | None,
) -> None:
    labels_by_id = load_text_evidence_labels(evidence_path)
    scorer = TextEvidenceCoverageScorer(metric_tokenizer=tokenizer)
    records = []
    for start in tqdm(
        range(0, len(query_records), bsz),
        total=(len(query_records) + bsz - 1) // bsz,
    ):
        batch_query_records = query_records[start : start + bsz]
        batch_queries = [record["query"] for record in batch_query_records]
        batch_docs = vectordb.find_top_k_docs(top_k=top_k, queries=batch_queries)
        compressed_batch_docs = (
            compress_docs(batch_queries, batch_docs, option=compress_method)
            if compress_method
            else batch_docs
        )

        for query_record, retrieved_docs, compressed_docs in zip(
            batch_query_records, batch_docs, compressed_batch_docs
        ):
            query = query_record["query"]
            label_id = query_record["id"]
            if isinstance(label_id, bool) or not isinstance(label_id, (int, str)):
                raise ValueError(
                    "ID-keyed evidence lookup requires every query row to have "
                    f"an integer or string ID; invalid ID for {query!r}: {label_id!r}"
                )
            label = labels_by_id.get(label_id)
            if label is None:
                raise KeyError(
                    f"query not found in text evidence file {evidence_path}: "
                    f"id={label_id!r}, query={query!r}"
                )
            if label.get("query") != query:
                raise ValueError(
                    f"query/evidence text mismatch for id={label_id!r}: "
                    f"{query!r} != {label.get('query')!r}"
                )
            retrieved_text = docs_full_text(retrieved_docs)
            compressed_text = (
                selected_context_text(compressed_docs)
                if compress_method
                else retrieved_text
            )
            score = scorer.score(
                label=label,
                retrieved_context=retrieved_text,
                compressed_context=compressed_text,
            )
            score["retrieval"]["context_tokens"] = actual_token_count(
                retrieved_text, tokenizer
            )
            score["retrieval"]["retrieved_chunks"] = len(retrieved_docs)
            score["compressed"]["context_tokens"] = actual_token_count(
                compressed_text, tokenizer
            )
            score["compressed"]["selected_subchunks"] = (
                selected_context_segment_count(compressed_docs)
                if compress_method
                else None
            )
            records.append(
                {
                    "id": label_id,
                    "source_id": label.get("source_id"),
                    "query": query,
                    "retrieved_context_text": retrieved_text,
                    "compressed_context_text": compressed_text,
                    **score,
                }
            )

    if len(records) != len(query_records):
        raise ValueError(
            f"evaluated record count differs from query count: "
            f"{len(records)} != {len(query_records)}"
        )
    summary = summarize_text_evidence_records(records)
    summary.update(
        {
            "dataset": dataset,
            "evidence_file": str(evidence_path),
            "query_file": query_file,
            "query_file_count": len(query_records),
            "db_dir": db_dir,
            "top_k": top_k,
            "compress_method": compress_method,
            "colbert_query": colbert_query_config,
            "metric_tokenizer_name": model_name,
            "model_name": model_name,
            "setup_time_sec": setup_time,
            "run_time_sec": time.perf_counter() - run_start,
        }
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if details_file:
        write_jsonl(Path(details_file), records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(
    dataset: str,
    query_file: str,
    db_dir: str,
    top_k: int,
    probe_query_limit: int | None,
    bsz: int,
    compress_method: str | None,
    evidence_file: str,
    model_name: str,
    output_file: str,
    details_file: str | None,
) -> None:
    logging.set_verbosity_error()
    evidence_path = Path(evidence_file)
    query_records = load_jsonl_query_records(
        Path(query_file), probe_query_limit=probe_query_limit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vectordb = ChromaDB(db_dir=db_dir)

    setup_start = time.perf_counter()
    active_compressor = (
        initialize_compressor(option=compress_method) if compress_method else None
    )
    active_encoder = getattr(active_compressor, "encoder", None)
    colbert_query_config = (
        {
            "maxlen": getattr(active_encoder, "query_maxlen", None),
            "minlen": getattr(active_encoder, "query_minlen", None),
            "truncation_side": getattr(active_encoder, "query_truncation_side", None),
        }
        if active_encoder is not None
        else None
    )
    setup_time = time.perf_counter() - setup_start

    run_start = time.perf_counter()
    run_retrieval_only_evaluation(
        dataset=dataset,
        query_records=query_records,
        vectordb=vectordb,
        tokenizer=tokenizer,
        query_file=query_file,
        db_dir=db_dir,
        evidence_path=evidence_path,
        top_k=top_k,
        bsz=bsz,
        compress_method=compress_method,
        model_name=model_name,
        setup_time=setup_time,
        colbert_query_config=colbert_query_config,
        run_start=run_start,
        output_file=output_file,
        details_file=details_file,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--probe_query_limit", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--compress_method", default=None)
    parser.add_argument("--evidence_file", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_file", default="outputs/retrieval-only-summary.json")
    parser.add_argument("--details_file", default=None)
    parsed = parser.parse_args()
    main(**vars(parsed))
