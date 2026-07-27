from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from itertools import islice
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


def load_jsonl_queries(path: Path, total_num: int) -> list[str]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, total_num):
            queries.append(parse_json_query(line))
    return queries


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
    queries: list[str],
    vectordb,
    tokenizer,
    db_dir: str,
    sample_path: Path,
    top_k: int,
    bsz: int,
    compress_method: str | None,
    model_name: str,
    passage_recall_threshold: float,
    setup_time: float,
    colbert_query_config: dict[str, Any] | None,
    run_start: float,
    output_file: str,
    details_file: str | None,
) -> None:
    labels_by_query = load_text_evidence_labels(sample_path)
    scorer = TextEvidenceCoverageScorer(
        metric_tokenizer=tokenizer,
        passage_recall_threshold=passage_recall_threshold,
    )
    records = []
    for start in tqdm(
        range(0, len(queries), bsz), total=(len(queries) + bsz - 1) // bsz
    ):
        batch_queries = queries[start : start + bsz]
        batch_docs = vectordb.find_top_k_docs(top_k=top_k, queries=batch_queries)
        compressed_batch_docs = (
            compress_docs(batch_queries, batch_docs, option=compress_method)
            if compress_method
            else batch_docs
        )

        for query, retrieved_docs, compressed_docs in zip(
            batch_queries, batch_docs, compressed_batch_docs
        ):
            label = labels_by_query.get(query)
            if label is None:
                raise KeyError(
                    f"query not found in text evidence file {sample_path}: "
                    f"{query!r}"
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
                    "id": len(records),
                    "source_id": label.get("source_id"),
                    "query": query,
                    **score,
                }
            )

    if len(records) != len(queries):
        raise ValueError(
            f"evaluated record count differs from query count: "
            f"{len(records)} != {len(queries)}"
        )
    summary = summarize_text_evidence_records(
        records,
        passage_recall_threshold=passage_recall_threshold,
    )
    summary.update(
        {
            "dataset": dataset,
            "sample_file": str(sample_path),
            "query_file_count": len(queries),
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
    total_num: int,
    bsz: int,
    compress_method: str | None,
    sample_file: str,
    model_name: str,
    output_file: str,
    details_file: str | None,
    passage_recall_threshold: float = 0.8,
) -> None:
    logging.set_verbosity_error()
    sample_path = Path(sample_file)
    queries = load_jsonl_queries(Path(query_file), total_num=total_num)
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
        queries=queries,
        vectordb=vectordb,
        tokenizer=tokenizer,
        db_dir=db_dir,
        sample_path=sample_path,
        top_k=top_k,
        bsz=bsz,
        compress_method=compress_method,
        model_name=model_name,
        passage_recall_threshold=passage_recall_threshold,
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
    parser.add_argument("--total_num", type=int, default=200)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--compress_method", default=None)
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--passage_recall_threshold", type=float, default=0.8)
    parser.add_argument("--output_file", default="outputs/retrieval-only-summary.json")
    parser.add_argument("--details_file", default=None)
    parsed = parser.parse_args()
    main(**vars(parsed))
