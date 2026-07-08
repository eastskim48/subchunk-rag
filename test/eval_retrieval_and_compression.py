from __future__ import annotations

import json
import os
import re
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

from compressor.factory import compress_docs, initialize_compressor
from utils import parse_json_query
from vectordb import ChromaDB

SUPPORTED_SAMPLE_FILES = {
    "longbench-hotpotqa": ROOT / "samples" / "hotpotqa.json",
    "hotpotqa": ROOT / "samples" / "hotpotqa.json",
    "longbench-2wiki": ROOT / "samples" / "2wiki.json",
    "2wiki": ROOT / "samples" / "2wiki.json",
}


def normalize_and_tokenize_for_overlap(text: str, tokenizer) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    token_ids = []
    for word in text.split(" "):
        if not word:
            continue
        token_ids.extend(tokenizer.encode(f" {word}", add_special_tokens=False))
    return [str(token_id) for token_id in token_ids]


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left

    previous = [0] * (len(short) + 1)
    for long_item in long:
        current = [0] * (len(short) + 1)
        for idx, short_item in enumerate(short, start=1):
            if long_item == short_item:
                current[idx] = previous[idx - 1] + 1
            else:
                current[idx] = max(previous[idx], current[idx - 1])
        previous = current
    return previous[-1]


def rouge_l_precision_recall(
    gold_tokens: list[str], selected_tokens: list[str]
) -> tuple[float, float]:
    if not gold_tokens or not selected_tokens:
        return 0.0, 0.0
    overlap = lcs_length(gold_tokens, selected_tokens)
    precision = overlap / len(selected_tokens) if selected_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    return precision, recall


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


def load_gold_by_query(
    sample_file: Path, gold_field: str = "supporting_facts"
) -> dict[str, list[str]]:
    samples = json.loads(sample_file.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"{sample_file} must contain a JSON list")

    gold_by_query = {}
    for sample in samples:
        query = sample.get("query")
        gold_passages = sample.get(gold_field)
        if not isinstance(query, str):
            continue
        if not isinstance(gold_passages, list):
            raise ValueError(
                f"{sample_file} sample for query {query!r} has no {gold_field} list"
            )
        gold_by_query[query] = [
            str(text) for text in gold_passages if isinstance(text, str)
        ]
    return gold_by_query


def _texts_from_gold_field(
    sample: dict[str, Any], field_name: str, query: str, sample_file: Path
) -> list[str]:
    value = sample.get(field_name)
    if not isinstance(value, list):
        raise ValueError(
            f"{sample_file} sample for query {query!r} has no {field_name} list"
        )

    texts = []
    for item in value:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            text = item.get("matched_text") or item.get("span_text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def load_eval_gold_by_query(
    sample_file: Path,
    rouge_gold_field: str = "supporting_facts",
    subchunk_gold_field: str = "supporting_sentences",
) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(sample_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        samples = payload["records"]
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError(
            f"{sample_file} must contain a JSON list or a dict with a records list"
        )

    gold_by_query = {}
    for sample in samples:
        query = sample.get("query")
        if not isinstance(query, str):
            continue

        rouge_passages = _texts_from_gold_field(
            sample, rouge_gold_field, query, sample_file
        )
        subchunk_passages = _texts_from_gold_field(
            sample, subchunk_gold_field, query, sample_file
        )

        supporting_fact_titles = sample.get("supporting_fact_titles")
        if not isinstance(supporting_fact_titles, list):
            raise ValueError(
                f"{sample_file} sample for query {query!r} has no supporting_fact_titles list"
            )

        gold_by_query[query] = {
            "rouge_passages": [
                str(text) for text in rouge_passages if isinstance(text, str)
            ],
            "subchunk_passages": [
                str(text) for text in subchunk_passages if isinstance(text, str)
            ],
            "supporting_fact_titles": [
                str(title) for title in supporting_fact_titles if isinstance(title, str)
            ],
        }
    return gold_by_query


def build_title_doc_id_map(db_dir: str) -> dict[str, set[str]]:
    docs_dir = Path(db_dir).parent.parent / "documents"
    if not docs_dir.exists():
        return {}

    doc_ids_by_title: dict[str, set[str]] = {}
    for path in sorted(docs_dir.glob("doc_*.txt")):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            title = handle.readline().strip()
        if title:
            doc_ids_by_title.setdefault(title, set()).add(path.name)
    return doc_ids_by_title


def docs_full_text(docs) -> str:
    return "\n\n".join(getattr(doc, "text", "") for doc in docs)


def docs_cacheable_text(docs) -> str:
    passages = []
    for doc in docs:
        for cacheable in getattr(doc, "cacheables", []) or []:
            passages.append(getattr(cacheable, "text", ""))
    return "\n\n".join(passages)


def docs_cacheable_count(docs) -> int:
    count = 0
    for doc in docs:
        for cacheable in getattr(doc, "cacheables", []) or []:
            count += 1
    return count


def docs_cacheable_ids(docs) -> list[str]:
    ids = []
    for doc in docs:
        for cacheable in getattr(doc, "cacheables", []) or []:
            ids.append(cacheable.id)
    return ids


def doc_parent_id(doc) -> str:
    doc_id = getattr(doc, "id", "")
    if isinstance(doc_id, str) and "::ret_" in doc_id:
        return doc_id.split("::ret_", 1)[0]
    metadata = getattr(doc, "metadata", {}) or {}
    parent_doc_id = metadata.get("parent_doc_id")
    if isinstance(parent_doc_id, str):
        return parent_doc_id
    cacheables = getattr(doc, "cacheables", []) or []
    for cacheable in cacheables:
        cacheable_parent_doc_id = getattr(cacheable, "parent_doc_id", None)
        if isinstance(cacheable_parent_doc_id, str):
            return cacheable_parent_doc_id
    return str(doc_id)


def find_min_gold_sentence_span(
    gold_fact: str,
    docs,
    tokenizer,
) -> dict[str, Any]:
    gold_tokens = normalize_and_tokenize_for_overlap(gold_fact, tokenizer)
    best = None

    for doc_idx, doc in enumerate(docs):
        cacheable_entries = [
            {
                "cacheable": cacheable,
                "text": cacheable.text,
                "overlap_tokens": normalize_and_tokenize_for_overlap(
                    cacheable.text, tokenizer
                ),
                "actual_tokens": actual_token_count(cacheable.text, tokenizer),
            }
            for cacheable in (getattr(doc, "cacheables", []) or [])
        ]
        flat_tokens = []
        token_to_cacheable_idx = []
        for cacheable_idx, entry in enumerate(cacheable_entries):
            flat_tokens.extend(entry["overlap_tokens"])
            token_to_cacheable_idx.extend(
                [cacheable_idx] * len(entry["overlap_tokens"])
            )

        if gold_tokens and flat_tokens and len(gold_tokens) <= len(flat_tokens):
            for token_start in range(0, len(flat_tokens) - len(gold_tokens) + 1):
                if (
                    flat_tokens[token_start : token_start + len(gold_tokens)]
                    != gold_tokens
                ):
                    continue
                start_idx = token_to_cacheable_idx[token_start]
                end_idx = token_to_cacheable_idx[token_start + len(gold_tokens) - 1]
                span_entries = cacheable_entries[start_idx : end_idx + 1]
                span_tokens = [
                    token for entry in span_entries for token in entry["overlap_tokens"]
                ]
                sentence_count = end_idx - start_idx + 1
                token_count = sum(entry["actual_tokens"] for entry in span_entries)
                precision, recall = rouge_l_precision_recall(gold_tokens, span_tokens)
                candidate = {
                    "doc_idx": doc_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "sentence_count": sentence_count,
                    "token_count": token_count,
                    "precision": precision,
                    "recall": recall,
                    "cacheables": [entry["cacheable"] for entry in span_entries],
                    "cacheable_ids": [entry["cacheable"].id for entry in span_entries],
                }
                if best is None or (
                    candidate["sentence_count"],
                    candidate["token_count"],
                    -candidate["precision"],
                ) < (
                    best["sentence_count"],
                    best["token_count"],
                    -best["precision"],
                ):
                    best = candidate

    return best or {
        "doc_idx": None,
        "start_idx": None,
        "end_idx": None,
        "sentence_count": 0,
        "token_count": 0,
        "precision": 0.0,
        "recall": 0.0,
        "cacheables": [],
        "cacheable_ids": [],
    }


def compressed_subchunk_level_metrics(
    gold_passages: list[str],
    selected_docs,
    tokenizer,
) -> dict[str, float]:
    selected_ids = docs_cacheable_ids(selected_docs)
    preserved_facts = 0
    gold_subchunk_ids = set()
    for gold_passage in gold_passages:
        match = find_min_gold_sentence_span(gold_passage, selected_docs, tokenizer)
        cacheable_ids = match.get("cacheable_ids") or []
        if not cacheable_ids:
            continue
        preserved_facts += 1
        gold_subchunk_ids.update(cacheable_ids)
    selected_gold_subchunks = sum(
        1 for cacheable_id in selected_ids if cacheable_id in gold_subchunk_ids
    )
    return {
        "subchunk_level_recall": (
            preserved_facts / len(gold_passages) if gold_passages else 0.0
        ),
        "subchunk_level_precision": (
            selected_gold_subchunks / len(selected_ids) if selected_ids else 0.0
        ),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def vanilla_chunk_level_metrics(
    supporting_fact_titles: list[str],
    title_doc_ids: dict[str, set[str]],
    vanilla_docs,
    top_k: int,
) -> dict[str, float]:
    gold_titles = list(dict.fromkeys(supporting_fact_titles))
    retrieved_doc_ids = {doc_parent_id(doc) for doc in vanilla_docs}
    matched_titles = 0
    matched_doc_ids = set()
    for title in gold_titles:
        candidate_doc_ids = title_doc_ids.get(title, set())
        matched_for_title = candidate_doc_ids & retrieved_doc_ids
        if matched_for_title:
            matched_titles += 1
            matched_doc_ids.update(matched_for_title)
    return {
        "chunk_level_recall": matched_titles / len(gold_titles) if gold_titles else 0.0,
        "chunk_level_precision": len(matched_doc_ids) / top_k if top_k else 0.0,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    vanilla_recall = mean([record["vanilla"]["rouge_l_recall"] for record in records])
    vanilla_precision = mean(
        [record["vanilla"]["rouge_l_precision"] for record in records]
    )
    vanilla_tokens = mean([record["vanilla"]["selected_tokens"] for record in records])
    vanilla_chunk_level_recall = mean(
        [record["vanilla"]["chunk_level_recall"] for record in records]
    )
    vanilla_chunk_level_precision = mean(
        [record["vanilla"]["chunk_level_precision"] for record in records]
    )
    vanilla_subchunk_level_recall = mean(
        [record["vanilla"]["subchunk_level_recall"] for record in records]
    )
    vanilla_subchunk_level_precision = mean(
        [record["vanilla"]["subchunk_level_precision"] for record in records]
    )
    compressed_recall = mean(
        [record["compressed"]["rouge_l_recall"] for record in records]
    )
    compressed_precision = mean(
        [record["compressed"]["rouge_l_precision"] for record in records]
    )
    compressed_tokens = mean(
        [record["compressed"]["token_counts"] for record in records]
    )
    compressed_subchunk_counts = mean(
        [record["compressed"]["subchunk_count"] for record in records]
    )
    compressed_subchunk_level_recall = mean(
        [record["compressed"]["subchunk_level_recall"] for record in records]
    )
    compressed_subchunk_level_precision = mean(
        [record["compressed"]["subchunk_level_precision"] for record in records]
    )

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
            "rouge_l_recall": vanilla_recall,
            "rouge_l_precision": vanilla_precision,
            "selected_tokens": vanilla_tokens,
            "chunk_level_recall": vanilla_chunk_level_recall,
            "chunk_level_precision": vanilla_chunk_level_precision,
            "subchunk_level_recall": vanilla_subchunk_level_recall,
            "subchunk_level_precision": vanilla_subchunk_level_precision,
        },
        "compressed": {
            "rouge_l_recall": compressed_recall,
            "rouge_l_precision": compressed_precision,
            "token_counts": compressed_tokens,
            "subchunk_counts": compressed_subchunk_counts,
            "subchunk_level_recall": compressed_subchunk_level_recall,
            "subchunk_level_precision": compressed_subchunk_level_precision,
        },
        "delta": {
            "recall_drop": vanilla_recall - compressed_recall,
            "precision_gain": compressed_precision - vanilla_precision,
            "token_reduction": vanilla_tokens - compressed_tokens,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_sample_file(dataset: str) -> Path:
    normalized = dataset.strip().lower()
    sample_file = SUPPORTED_SAMPLE_FILES.get(normalized)
    if sample_file is None:
        raise ValueError(
            f"no default gold sample file for dataset={dataset!r}; "
            "pass --sample_file explicitly"
        )
    return sample_file


def main(
    dataset: str,
    query_file: str,
    db_dir: str,
    top_k: int,
    total_num: int,
    bsz: int,
    compress_method: str | None,
    sample_file: str | None,
    gold_field: str,
    model_name: str,
    output_file: str,
    details_file: str | None,
) -> None:
    logging.set_verbosity_error()
    sample_path = Path(sample_file) if sample_file else default_sample_file(dataset)
    subchunk_gold_field = (
        "llm_valid_evidence_subchunk_texts"
        if gold_field == "llm_valid_evidence_spans"
        else "supporting_sentences"
    )
    gold_by_query = load_eval_gold_by_query(
        sample_path,
        rouge_gold_field=gold_field,
        subchunk_gold_field=subchunk_gold_field,
    )
    title_doc_ids = build_title_doc_id_map(db_dir)
    queries = load_jsonl_queries(Path(query_file), total_num=total_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vectordb = ChromaDB(db_dir=db_dir)

    setup_start = time.perf_counter()
    if compress_method:
        initialize_compressor(option=compress_method)
    setup_time = time.perf_counter() - setup_start

    records = []
    run_start = time.perf_counter()
    for start in tqdm(
        range(0, len(queries), bsz), total=(len(queries) + bsz - 1) // bsz
    ):
        batch_queries = queries[start : start + bsz]
        batch_docs = vectordb.find_top_k_docs(top_k=top_k, queries=batch_queries)
        if compress_method:
            compressed_batch_docs = compress_docs(
                batch_queries, batch_docs, option=compress_method
            )
        else:
            compressed_batch_docs = batch_docs

        for query, vanilla_docs, compressed_docs in zip(
            batch_queries, batch_docs, compressed_batch_docs
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
            vanilla_text = docs_full_text(vanilla_docs)
            if compress_method:
                compressed_text = docs_cacheable_text(compressed_docs)
            else:
                compressed_text = vanilla_text

            gold_overlap_tokens = normalize_and_tokenize_for_overlap(
                gold_text, tokenizer
            )
            vanilla_overlap_tokens = normalize_and_tokenize_for_overlap(
                vanilla_text, tokenizer
            )
            compressed_overlap_tokens = normalize_and_tokenize_for_overlap(
                compressed_text, tokenizer
            )
            vanilla_precision, vanilla_recall = rouge_l_precision_recall(
                gold_tokens=gold_overlap_tokens,
                selected_tokens=vanilla_overlap_tokens,
            )
            compressed_precision, compressed_recall = rouge_l_precision_recall(
                gold_tokens=gold_overlap_tokens,
                selected_tokens=compressed_overlap_tokens,
            )
            gold_tokens = actual_token_count(gold_text, tokenizer)
            vanilla_tokens = actual_token_count(vanilla_text, tokenizer)
            compressed_tokens = actual_token_count(compressed_text, tokenizer)
            compressed_subchunk_count = docs_cacheable_count(compressed_docs)
            supporting_fact_title_counts = len(
                list(dict.fromkeys(supporting_fact_titles))
            )
            vanilla_chunk_metrics = vanilla_chunk_level_metrics(
                supporting_fact_titles=supporting_fact_titles,
                title_doc_ids=title_doc_ids,
                vanilla_docs=vanilla_docs,
                top_k=top_k,
            )
            vanilla_subchunk_metrics = compressed_subchunk_level_metrics(
                gold_passages=subchunk_gold_passages,
                selected_docs=vanilla_docs,
                tokenizer=tokenizer,
            )
            compressed_subchunk_metrics = compressed_subchunk_level_metrics(
                gold_passages=subchunk_gold_passages,
                selected_docs=compressed_docs,
                tokenizer=tokenizer,
            )

            records.append(
                {
                    "id": sample_idx,
                    "query": query,
                    "gold": {
                        "tokens": gold_tokens,
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
                        "selected_tokens": vanilla_tokens,
                        "chunk_level_recall": vanilla_chunk_metrics[
                            "chunk_level_recall"
                        ],
                        "chunk_level_precision": vanilla_chunk_metrics[
                            "chunk_level_precision"
                        ],
                        "subchunk_level_recall": vanilla_subchunk_metrics[
                            "subchunk_level_recall"
                        ],
                        "subchunk_level_precision": vanilla_subchunk_metrics[
                            "subchunk_level_precision"
                        ],
                        "retrieved_docs": len(vanilla_docs),
                    },
                    "compressed": {
                        "rouge_l_recall": compressed_recall,
                        "rouge_l_precision": compressed_precision,
                        "token_counts": compressed_tokens,
                        "subchunk_count": compressed_subchunk_count,
                        "subchunk_level_recall": compressed_subchunk_metrics[
                            "subchunk_level_recall"
                        ],
                        "subchunk_level_precision": compressed_subchunk_metrics[
                            "subchunk_level_precision"
                        ],
                        "retrieved_docs": len(compressed_docs),
                    },
                    "delta": {
                        "recall_drop": vanilla_recall - compressed_recall,
                        "precision_gain": compressed_precision - vanilla_precision,
                        "token_reduction": vanilla_tokens - compressed_tokens,
                    },
                }
            )

    summary = summarize(records)
    summary.update(
        {
            "dataset": dataset,
            "sample_file": str(sample_path),
            "gold_field": gold_field,
            "rouge_gold_field": gold_field,
            "subchunk_level_gold_field": subchunk_gold_field,
            "chunk_level_gold_field": "supporting_fact_titles",
            "query_file": query_file,
            "db_dir": db_dir,
            "top_k": top_k,
            "compress_method": compress_method,
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
    parser.add_argument("--compress_method", default=None)
    parser.add_argument("--sample_file", default=None)
    parser.add_argument("--gold_field", default="supporting_facts")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--output_file", default="outputs/eval-retrieval-and-compression-summary.json"
    )
    parser.add_argument("--details_file", default=None)
    parsed = parser.parse_args()
    main(**vars(parsed))
