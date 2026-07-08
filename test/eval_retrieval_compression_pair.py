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

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TEST_DIR = ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from compressor.factory import COMPRESSOR_TYPES  # noqa: E402
from eval_retrieval_and_compression import (  # noqa: E402
    SUPPORTED_SAMPLE_FILES,
    actual_token_count,
    compressed_subchunk_level_metrics,
    default_sample_file,
    docs_cacheable_count,
    docs_cacheable_ids,
    docs_cacheable_text,
    docs_full_text,
    build_title_doc_id_map,
    find_min_gold_sentence_span,
    load_eval_gold_by_query,
    mean,
    normalize_and_tokenize_for_overlap,
    rouge_l_precision_recall,
    vanilla_chunk_level_metrics,
    write_jsonl,
)
from utils import parse_json_query  # noqa: E402
from vectordb import ChromaDB  # noqa: E402


def load_jsonl_queries(path: Path, total_num: int) -> list[str]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, total_num):
            queries.append(parse_json_query(line))
    return queries


def normalize_method(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "" or value == "none":
        return None
    return value


def build_compressor(method: str | None):
    if method is None:
        return None
    compressor_type = COMPRESSOR_TYPES.get(method)
    if compressor_type is None:
        raise ValueError(f"Unknown compression option: {method}")
    return compressor_type()


def compress_with(compressor, batch_queries, batch_docs):
    if compressor is None:
        return batch_docs
    return compressor.compress_batch_top_k_docs(
        batch_top_k_docs=batch_docs,
        batch_queries=batch_queries,
    )


def selected_context_text(docs, method: str | None) -> str:
    if method is None:
        return docs_full_text(docs)
    return docs_cacheable_text(docs)


def method_metrics(
    docs,
    method: str | None,
    gold_overlap_tokens: list[str],
    subchunk_gold_passages: list[str],
    tokenizer,
) -> dict[str, Any]:
    text = selected_context_text(docs, method)
    selected_overlap_tokens = normalize_and_tokenize_for_overlap(text, tokenizer)
    precision, recall = rouge_l_precision_recall(
        gold_tokens=gold_overlap_tokens,
        selected_tokens=selected_overlap_tokens,
    )
    subchunk_metrics = compressed_subchunk_level_metrics(
        gold_passages=subchunk_gold_passages,
        selected_docs=docs,
        tokenizer=tokenizer,
    )
    return {
        "rouge_l_recall": recall,
        "rouge_l_precision": precision,
        "token_counts": actual_token_count(text, tokenizer),
        "subchunk_counts": docs_cacheable_count(docs),
        "subchunk_level_recall": subchunk_metrics["subchunk_level_recall"],
        "subchunk_level_precision": subchunk_metrics["subchunk_level_precision"],
    }


def compare_selected_subchunks(
    method_a_docs,
    method_b_docs,
    subchunk_gold_passages: list[str],
    tokenizer,
) -> dict[str, float]:
    a_ids = set(docs_cacheable_ids(method_a_docs))
    b_ids = set(docs_cacheable_ids(method_b_docs))
    text_by_id = {}
    for docs in (method_a_docs, method_b_docs):
        for doc in docs:
            for cacheable in getattr(doc, "cacheables", []) or []:
                text_by_id.setdefault(cacheable.id, getattr(cacheable, "text", ""))

    intersection = a_ids & b_ids
    union = a_ids | b_ids
    intersection_token_count = sum(
        actual_token_count(text_by_id[cacheable_id], tokenizer)
        for cacheable_id in intersection
    )
    union_token_count = sum(
        actual_token_count(text_by_id[cacheable_id], tokenizer)
        for cacheable_id in union
    )
    gold_both = 0
    gold_a_only = 0
    gold_b_only = 0
    gold_neither = 0
    for gold_passage in subchunk_gold_passages:
        a_match = find_min_gold_sentence_span(gold_passage, method_a_docs, tokenizer)
        b_match = find_min_gold_sentence_span(gold_passage, method_b_docs, tokenizer)
        a_hit = bool(a_match.get("cacheable_ids") or [])
        b_hit = bool(b_match.get("cacheable_ids") or [])
        if a_hit and b_hit:
            gold_both += 1
        elif a_hit:
            gold_a_only += 1
        elif b_hit:
            gold_b_only += 1
        else:
            gold_neither += 1

    gold_count = len(subchunk_gold_passages)
    return {
        "intersection_token_counts": float(intersection_token_count),
        "union_token_counts": float(union_token_count),
        "intersection_subchunk_counts": float(len(intersection)),
        "union_subchunk_counts": float(len(union)),
        "both_recall": gold_both / gold_count if gold_count else 0.0,
        "a_only_recall": gold_a_only / gold_count if gold_count else 0.0,
        "b_only_recall": gold_b_only / gold_count if gold_count else 0.0,
        "neither_recall": gold_neither / gold_count if gold_count else 0.0,
    }


def summarize_method(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    fields = [
        "rouge_l_recall",
        "rouge_l_precision",
        "token_counts",
        "subchunk_counts",
        "subchunk_level_recall",
        "subchunk_level_precision",
    ]
    return {field: mean([record[key][field] for record in records]) for field in fields}


def summarize_pair(records: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "intersection_token_counts",
        "union_token_counts",
        "intersection_subchunk_counts",
        "union_subchunk_counts",
        "both_recall",
        "a_only_recall",
        "b_only_recall",
        "neither_recall",
    ]
    return {
        field: mean([record["pair"][field] for record in records]) for field in fields
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "gold": {
            "tokens": mean([record["gold"]["tokens"] for record in records]),
            "chunk_counts": mean(
                [record["gold"]["chunk_counts"] for record in records]
            ),
            "subchunk_counts": mean(
                [record["gold"]["subchunk_counts"] for record in records]
            ),
            "subchunk_rouge_l_recall": mean(
                [record["gold"]["subchunk_rouge_l_recall"] for record in records]
            ),
        },
        "vanilla": {
            "rouge_l_recall": mean(
                [record["vanilla"]["rouge_l_recall"] for record in records]
            ),
            "rouge_l_precision": mean(
                [record["vanilla"]["rouge_l_precision"] for record in records]
            ),
            "selected_tokens": mean(
                [record["vanilla"]["selected_tokens"] for record in records]
            ),
            "chunk_level_recall": mean(
                [record["vanilla"]["chunk_level_recall"] for record in records]
            ),
            "chunk_level_precision": mean(
                [record["vanilla"]["chunk_level_precision"] for record in records]
            ),
        },
        "method_a": summarize_method(records, "method_a"),
        "method_b": summarize_method(records, "method_b"),
        "pair": summarize_pair(records),
    }


def main(
    dataset: str,
    query_file: str,
    db_dir: str,
    top_k: int,
    total_num: int,
    bsz: int,
    method_a: str | None,
    method_b: str | None,
    sample_file: str | None,
    gold_field: str,
    model_name: str,
    output_file: str,
    details_file: str | None,
) -> None:
    logging.set_verbosity_error()
    method_a = normalize_method(method_a)
    method_b = normalize_method(method_b)
    sample_path = Path(sample_file) if sample_file else default_sample_file(dataset)
    gold_by_query = load_eval_gold_by_query(
        sample_path,
        rouge_gold_field=gold_field,
        subchunk_gold_field="supporting_sentences",
    )
    title_doc_ids = build_title_doc_id_map(db_dir)
    queries = load_jsonl_queries(Path(query_file), total_num=total_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vectordb = ChromaDB(db_dir=db_dir)

    setup_start = time.perf_counter()
    compressor_a = build_compressor(method_a)
    compressor_b = build_compressor(method_b)
    setup_time = time.perf_counter() - setup_start

    records = []
    run_start = time.perf_counter()
    for start in tqdm(
        range(0, len(queries), bsz), total=(len(queries) + bsz - 1) // bsz
    ):
        batch_queries = queries[start : start + bsz]
        batch_docs = vectordb.find_top_k_docs(top_k=top_k, queries=batch_queries)
        method_a_docs = compress_with(compressor_a, batch_queries, batch_docs)
        method_b_docs = compress_with(compressor_b, batch_queries, batch_docs)

        for query, vanilla_docs, docs_a, docs_b in zip(
            batch_queries, batch_docs, method_a_docs, method_b_docs
        ):
            sample_idx = len(records)
            gold_record = gold_by_query.get(query)
            if gold_record is None:
                raise KeyError(
                    f"query not found in gold sample file {sample_path}: {query!r}"
                )
            rouge_gold_passages = gold_record["rouge_passages"]
            subchunk_gold_passages = gold_record["subchunk_passages"]
            supporting_fact_titles = gold_record["supporting_fact_titles"]

            gold_text = "\n\n".join(rouge_gold_passages)
            gold_overlap_tokens = normalize_and_tokenize_for_overlap(
                gold_text, tokenizer
            )
            vanilla_text = docs_full_text(vanilla_docs)
            vanilla_overlap_tokens = normalize_and_tokenize_for_overlap(
                vanilla_text, tokenizer
            )
            vanilla_precision, vanilla_recall = rouge_l_precision_recall(
                gold_tokens=gold_overlap_tokens,
                selected_tokens=vanilla_overlap_tokens,
            )
            supporting_fact_title_counts = len(
                list(dict.fromkeys(supporting_fact_titles))
            )
            vanilla_chunk_metrics = vanilla_chunk_level_metrics(
                supporting_fact_titles=supporting_fact_titles,
                title_doc_ids=title_doc_ids,
                vanilla_docs=vanilla_docs,
                top_k=top_k,
            )

            records.append(
                {
                    "id": sample_idx,
                    "query": query,
                    "gold": {
                        "tokens": actual_token_count(gold_text, tokenizer),
                        "passage_count": len(rouge_gold_passages),
                        "subchunk_gold_passage_count": len(subchunk_gold_passages),
                        "supporting_fact_title_count": len(supporting_fact_titles),
                        "chunk_counts": supporting_fact_title_counts,
                        "subchunk_counts": len(rouge_gold_passages),
                        "subchunk_rouge_l_recall": 1.0,
                    },
                    "vanilla": {
                        "rouge_l_recall": vanilla_recall,
                        "rouge_l_precision": vanilla_precision,
                        "selected_tokens": actual_token_count(vanilla_text, tokenizer),
                        "chunk_level_recall": vanilla_chunk_metrics[
                            "chunk_level_recall"
                        ],
                        "chunk_level_precision": vanilla_chunk_metrics[
                            "chunk_level_precision"
                        ],
                        "retrieved_docs": len(vanilla_docs),
                    },
                    "method_a": method_metrics(
                        docs=docs_a,
                        method=method_a,
                        gold_overlap_tokens=gold_overlap_tokens,
                        subchunk_gold_passages=subchunk_gold_passages,
                        tokenizer=tokenizer,
                    ),
                    "method_b": method_metrics(
                        docs=docs_b,
                        method=method_b,
                        gold_overlap_tokens=gold_overlap_tokens,
                        subchunk_gold_passages=subchunk_gold_passages,
                        tokenizer=tokenizer,
                    ),
                    "pair": compare_selected_subchunks(
                        method_a_docs=docs_a,
                        method_b_docs=docs_b,
                        subchunk_gold_passages=subchunk_gold_passages,
                        tokenizer=tokenizer,
                    ),
                }
            )

    summary = summarize(records)
    summary.update(
        {
            "dataset": dataset,
            "sample_file": str(sample_path),
            "gold_field": gold_field,
            "rouge_gold_field": gold_field,
            "subchunk_level_gold_field": "supporting_sentences",
            "chunk_level_gold_field": "supporting_fact_titles",
            "query_file": query_file,
            "db_dir": db_dir,
            "top_k": top_k,
            "method_a_name": method_a,
            "method_b_name": method_b,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--total_num", type=int, default=200)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--method_a", required=True)
    parser.add_argument("--method_b", required=True)
    parser.add_argument("--sample_file", default=None)
    parser.add_argument("--gold_field", default="supporting_facts")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--output_file", default="outputs/eval-retrieval-compression-pair-summary.json"
    )
    parser.add_argument("--details_file", default=None)
    parsed = parser.parse_args()
    main(**vars(parsed))
