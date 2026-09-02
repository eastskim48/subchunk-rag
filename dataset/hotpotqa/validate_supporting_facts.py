#!/usr/bin/env python
"""Validate HotpotQA supporting facts in the custom full-document corpus."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

LINK_TAG = re.compile(r"</?a(?:\s+[^>]*)?>", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def resolution_page_ids(resolution: dict[str, Any]) -> list[str]:
    status = resolution["status"]
    match_mode = resolution["match_mode"]
    if status == "matched":
        return [resolution["matched_page_id"]]
    if status == "ambiguous" and match_mode == "casefold_exact":
        return list(resolution["candidate_page_ids"])
    return []


def build_title_indexes(
    *,
    documents: list[dict[str, Any]],
    title_joins: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    document_by_id = {}
    for document in documents:
        page_id = str(document["page_id"])
        if page_id in document_by_id:
            raise ValueError(f"duplicate materialized page ID: {page_id}")
        document_by_id[page_id] = document

    title_to_documents = {}
    for resolution in title_joins:
        title = resolution["requested_title"]
        if title in title_to_documents:
            raise ValueError(f"duplicate title resolution: {title!r}")
        values = []
        for page_id in resolution_page_ids(resolution):
            document = document_by_id.get(str(page_id))
            if document is None:
                raise ValueError(
                    f"title {title!r} references missing materialized page {page_id}"
                )
            values.append(
                {
                    "page_id": str(page_id),
                    "wikipedia_title": document["title"],
                    "document_file": document["document_file"],
                }
            )
        title_to_documents[title] = values

    document_file_to_title = {}
    for document in documents:
        document_file = document["document_file"]
        if document_file in document_file_to_title:
            raise ValueError(f"duplicate document file: {document_file}")
        document_file_to_title[document_file] = {
            "page_id": str(document["page_id"]),
            "wikipedia_title": document["title"],
        }
    return title_to_documents, document_file_to_title


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def supporting_sentence_plaintext(text: str) -> str:
    return html.unescape(LINK_TAG.sub("", text))


def validate_supporting_facts(
    *,
    dataset_root: Path,
    dev_records: list[dict[str, Any]],
    title_to_documents: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    @lru_cache(maxsize=None)
    def load_document(relative_path: str) -> str:
        return (dataset_root / relative_path).read_text(encoding="utf-8")

    results = []
    fact_status_counts: Counter[str] = Counter()
    exact_query_count = 0
    normalized_query_count = 0
    total_fact_count = 0
    exact_fact_count = 0
    normalized_fact_count = 0

    for record in dev_records:
        example_id = record["_id"]
        context_by_title = {}
        duplicate_context_titles = set()
        for title, sentences in record["context"]:
            if title in context_by_title:
                duplicate_context_titles.add(title)
            else:
                context_by_title[title] = sentences

        fact_results = []
        for title, sentence_index in record["supporting_facts"]:
            total_fact_count += 1
            base = {
                "title": title,
                "sentence_index": sentence_index,
                "sentence_text": None,
                "comparison_sentence_text": None,
                "candidate_page_ids": [],
                "candidate_document_files": [],
                "trimmed_exact_match_page_ids": [],
                "whitespace_normalized_match_page_ids": [],
            }
            if title in duplicate_context_titles:
                base["status"] = "duplicate_context_title"
                fact_results.append(base)
                fact_status_counts[base["status"]] += 1
                continue
            sentences = context_by_title.get(title)
            if sentences is None:
                base["status"] = "missing_context_title"
                fact_results.append(base)
                fact_status_counts[base["status"]] += 1
                continue
            if (
                isinstance(sentence_index, bool)
                or not isinstance(sentence_index, int)
                or sentence_index < 0
                or sentence_index >= len(sentences)
            ):
                base["status"] = "invalid_sentence_index"
                fact_results.append(base)
                fact_status_counts[base["status"]] += 1
                continue

            sentence = sentences[sentence_index]
            comparison_sentence = supporting_sentence_plaintext(sentence)
            trimmed_sentence = comparison_sentence.strip()
            normalized_sentence = normalize_whitespace(comparison_sentence)
            candidate_documents = title_to_documents.get(title, [])
            exact_matches = []
            normalized_matches = []
            for document in candidate_documents:
                text = load_document(document["document_file"])
                if trimmed_sentence in text:
                    exact_matches.append(document["page_id"])
                elif normalized_sentence in normalize_whitespace(text):
                    normalized_matches.append(document["page_id"])

            base.update(
                {
                    "sentence_text": sentence,
                    "comparison_sentence_text": comparison_sentence,
                    "candidate_page_ids": [
                        document["page_id"] for document in candidate_documents
                    ],
                    "candidate_document_files": [
                        document["document_file"] for document in candidate_documents
                    ],
                    "trimmed_exact_match_page_ids": exact_matches,
                    "whitespace_normalized_match_page_ids": normalized_matches,
                }
            )
            if exact_matches:
                base["status"] = "trimmed_exact_match"
                exact_fact_count += 1
                normalized_fact_count += 1
            elif normalized_matches:
                base["status"] = "whitespace_normalized_match_only"
                normalized_fact_count += 1
            else:
                base["status"] = "not_found_in_document"
            fact_results.append(base)
            fact_status_counts[base["status"]] += 1

        all_exact = bool(fact_results) and all(
            fact["status"] == "trimmed_exact_match" for fact in fact_results
        )
        all_normalized = bool(fact_results) and all(
            fact["status"]
            in {"trimmed_exact_match", "whitespace_normalized_match_only"}
            for fact in fact_results
        )
        exact_query_count += int(all_exact)
        normalized_query_count += int(all_normalized)
        results.append(
            {
                "_id": example_id,
                "supporting_fact_count": len(fact_results),
                "all_supporting_facts_trimmed_exact": all_exact,
                "all_supporting_facts_whitespace_normalized": all_normalized,
                "supporting_facts": fact_results,
            }
        )

    query_count = len(dev_records)
    summary = {
        "validation_protocol": {
            "title_lookup": "verbatim HotpotQA title lookup in title_to_documents",
            "sentence_lookup": "verbatim context sentence at supporting-fact index",
            "sentence_serialization": (
                "remove HTML anchor tags and unescape HTML entities exactly as for "
                "the stored document plaintext"
            ),
            "primary_match": (
                "sentence boundary whitespace stripped, then case-sensitive "
                "contiguous substring search"
            ),
            "secondary_match": (
                "case-sensitive contiguous substring search after whitespace normalization"
            ),
            "fuzzy_matching": False,
        },
        "query_count": query_count,
        "supporting_fact_count": total_fact_count,
        "fact_status_counts": dict(sorted(fact_status_counts.items())),
        "trimmed_exact_fact_count": exact_fact_count,
        "trimmed_exact_fact_rate": (
            exact_fact_count / total_fact_count if total_fact_count else 0.0
        ),
        "whitespace_normalized_fact_count": normalized_fact_count,
        "whitespace_normalized_fact_rate": (
            normalized_fact_count / total_fact_count if total_fact_count else 0.0
        ),
        "all_facts_trimmed_exact_query_count": exact_query_count,
        "all_facts_trimmed_exact_query_rate": (
            exact_query_count / query_count if query_count else 0.0
        ),
        "all_facts_whitespace_normalized_query_count": normalized_query_count,
        "all_facts_whitespace_normalized_query_rate": (
            normalized_query_count / query_count if query_count else 0.0
        ),
    }
    return results, summary


def validate_dataset(
    dataset_root: Path,
    *,
    documents_index_path: Path | None = None,
) -> dict[str, Any]:
    """Validate supporting facts and update the dataset manifest."""

    dataset_info = dataset_root / "dataset_info"
    dev_path = dataset_root / "hotpot_dev_distractor_v1.json"
    documents_index_path = documents_index_path or dataset_info / "documents.jsonl"
    documents = load_jsonl(documents_index_path)
    title_joins = load_jsonl(dataset_info / "title_join.jsonl")
    with dev_path.open("r", encoding="utf-8") as handle:
        dev_records = json.load(handle)
    title_to_documents, document_file_to_title = build_title_indexes(
        documents=documents, title_joins=title_joins
    )
    validation_records, summary = validate_supporting_facts(
        dataset_root=dataset_root,
        dev_records=dev_records,
        title_to_documents=title_to_documents,
    )
    write_json_atomic(dataset_info / "title_to_documents.json", title_to_documents)
    write_json_atomic(
        dataset_info / "document_file_to_title.json", document_file_to_title
    )
    write_jsonl_atomic(
        dataset_info / "supporting_fact_validation.jsonl", validation_records
    )
    write_json_atomic(dataset_info / "supporting_fact_validation_summary.json", summary)

    manifest_path = dataset_info / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["derived_artifacts"] = {
        "title_to_documents": "dataset_info/title_to_documents.json",
        "document_file_to_title": "dataset_info/document_file_to_title.json",
        "supporting_fact_validation": "dataset_info/supporting_fact_validation.jsonl",
        "supporting_fact_validation_summary": (
            "dataset_info/supporting_fact_validation_summary.json"
        ),
    }
    manifest["supporting_fact_validation"] = {
        "supporting_fact_count": summary["supporting_fact_count"],
        "trimmed_exact_match_count": summary["trimmed_exact_fact_count"],
        "whitespace_normalized_match_only_count": summary["fact_status_counts"].get(
            "whitespace_normalized_match_only", 0
        ),
        "source_annotation_issue_count": summary["supporting_fact_count"]
        - summary["whitespace_normalized_fact_count"],
        "fuzzy_matching": False,
    }
    if documents_index_path != dataset_info / "documents.jsonl":
        manifest["supporting_fact_validation"]["document_index"] = str(
            documents_index_path.relative_to(dataset_root)
        )
    write_json_atomic(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    summary = validate_dataset(args.dataset_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
