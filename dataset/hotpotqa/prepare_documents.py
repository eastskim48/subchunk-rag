#!/usr/bin/env python
"""Build the custom HotpotQA distractor-dev full-Wikipedia corpus."""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import hashlib
import html
import io
import json
import math
import re
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

OFFICIAL_DEV_URL = (
    "https://huggingface.co/datasets/RAGLAB/data/resolve/"
    "c33b09b7fa5099b89830e84f9bdff78329268d9b/"
    "eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json?download=true"
)
OFFICIAL_WIKI_URL = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
)
EXPECTED_DEV_SIZE = 46_320_117
EXPECTED_DEV_SHA256 = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
EXPECTED_WIKI_SIZE = 7_413_895_794
EXPECTED_WIKI_MD5 = "62b8027b5803173d4383669d8d162509"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
LINK_TAG = re.compile(r"</?a(?:\s+[^>]*)?>", re.IGNORECASE)

_WORKER_REQUESTED_CASEFOLD: set[str] = set()
_WORKER_REQUESTED_COMPACT: set[str] = set()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def casefold_title(title: str) -> str:
    """Return the official case-insensitive title key after Unicode NFKC."""

    return unicodedata.normalize("NFKC", title).casefold()


def compact_title(title: str) -> str:
    """Remove whitespace, punctuation, and symbols without fuzzy matching."""

    return "".join(
        character for character in casefold_title(title) if character.isalnum()
    )


def load_dev_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain one JSON array")

    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"dev record {index} is not an object")
        example_id = record.get("_id")
        contexts = record.get("context")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"dev record {index} has invalid _id")
        if example_id in seen_ids:
            raise ValueError(f"duplicate dev _id: {example_id}")
        seen_ids.add(example_id)
        if not isinstance(contexts, list):
            raise ValueError(f"dev record {example_id} has invalid context")
        for context_index, context in enumerate(contexts):
            if not (
                isinstance(context, list)
                and len(context) == 2
                and isinstance(context[0], str)
                and isinstance(context[1], list)
            ):
                raise ValueError(
                    f"dev record {example_id} has invalid context {context_index}"
                )
    return records


def collect_requested_titles(
    records: list[dict[str, Any]],
) -> tuple[list[str], Counter[str]]:
    title_occurrences: Counter[str] = Counter()
    for record in records:
        title_occurrences.update(context[0] for context in record["context"])
    return (
        sorted(title_occurrences, key=lambda title: (casefold_title(title), title)),
        title_occurrences,
    )


def initialize_scan_worker(
    requested_casefold: set[str], requested_compact: set[str]
) -> None:
    global _WORKER_REQUESTED_CASEFOLD, _WORKER_REQUESTED_COMPACT
    _WORKER_REQUESTED_CASEFOLD = requested_casefold
    _WORKER_REQUESTED_COMPACT = requested_compact


def validate_wikipedia_page(page: dict[str, Any]) -> None:
    if not isinstance(page.get("id"), (int, str)):
        raise ValueError("Wikipedia page has invalid id")
    if not isinstance(page.get("title"), str) or not page["title"]:
        raise ValueError("Wikipedia page has invalid title")
    text = page.get("text")
    if not isinstance(text, list):
        raise ValueError(f"Wikipedia page {page['id']} has invalid text")
    for paragraph in text:
        if not (
            isinstance(paragraph, list)
            and all(isinstance(sentence, str) for sentence in paragraph)
        ):
            raise ValueError(f"Wikipedia page {page['id']} has invalid paragraph")


def scan_wikipedia_shard(
    member_name: str, compressed_shard: bytes
) -> tuple[int, list[dict[str, Any]]]:
    page_count = 0
    candidates = []
    with bz2.open(io.BytesIO(compressed_shard), mode="rt", encoding="utf-8") as shard:
        for line_number, line in enumerate(shard, start=1):
            if not line.strip():
                continue
            page = json.loads(line)
            if not isinstance(page, dict):
                raise ValueError(f"{member_name}:{line_number} is not a JSON object")
            page_count += 1
            title = page.get("title")
            if not isinstance(title, str):
                continue
            if (
                casefold_title(title) not in _WORKER_REQUESTED_CASEFOLD
                and compact_title(title) not in _WORKER_REQUESTED_COMPACT
            ):
                continue
            validate_wikipedia_page(page)
            candidates.append(page)
    return page_count, candidates


def scan_candidate_pages(
    *,
    archive_path: Path,
    requested_titles: list[str],
    candidate_path: Path,
    scan_workers: int,
) -> dict[str, Any]:
    requested_casefold = {casefold_title(title) for title in requested_titles}
    requested_compact = {compact_title(title) for title in requested_titles}
    by_casefold: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_compact: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    page_count = 0
    candidate_count = 0
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    def consume_result(result, output) -> None:
        nonlocal page_count, candidate_count
        shard_page_count, pages = result
        page_count += shard_page_count
        for page in pages:
            encoded = (
                json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            offset = output.tell()
            output.write(encoded)
            candidate = {
                "page_id": str(page["id"]),
                "title": page["title"],
                "offset": offset,
                "length": len(encoded),
            }
            by_casefold[casefold_title(page["title"])].append(candidate)
            by_compact[compact_title(page["title"])].append(candidate)
            candidate_count += 1

    with (
        candidate_path.open("wb") as output,
        tarfile.open(archive_path, mode="r:bz2") as archive,
    ):
        if scan_workers == 1:
            initialize_scan_worker(requested_casefold, requested_compact)
            for member in archive:
                if not member.isfile():
                    continue
                member_handle = archive.extractfile(member)
                if member_handle is None:
                    raise ValueError(f"cannot read Wikipedia shard: {member.name}")
                consume_result(
                    scan_wikipedia_shard(member.name, member_handle.read()), output
                )
        else:
            max_pending = max(1, scan_workers * 2)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=scan_workers,
                initializer=initialize_scan_worker,
                initargs=(requested_casefold, requested_compact),
            ) as executor:
                pending: list[concurrent.futures.Future] = []
                for member in archive:
                    if not member.isfile():
                        continue
                    member_handle = archive.extractfile(member)
                    if member_handle is None:
                        raise ValueError(f"cannot read Wikipedia shard: {member.name}")
                    pending.append(
                        executor.submit(
                            scan_wikipedia_shard, member.name, member_handle.read()
                        )
                    )
                    if len(pending) >= max_pending:
                        consume_result(pending.pop(0).result(), output)
                for future in pending:
                    consume_result(future.result(), output)

    return {
        "page_count": page_count,
        "candidate_count": candidate_count,
        "by_casefold": dict(by_casefold),
        "by_compact": dict(by_compact),
    }


def distinct_page_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {}
    for candidate in candidates:
        page_id = candidate["page_id"]
        previous = by_id.get(page_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"conflicting candidate for page {page_id}")
        by_id[page_id] = candidate
    return sorted(by_id.values(), key=lambda value: int(value["page_id"]))


def resolve_titles(
    *,
    requested_titles: list[str],
    title_occurrences: Counter[str],
    by_casefold: dict[str, list[dict[str, Any]]],
    by_compact: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    resolutions = []
    for title in requested_titles:
        casefold_candidates = distinct_page_candidates(
            by_casefold.get(casefold_title(title), [])
        )
        if len(casefold_candidates) == 1:
            candidate = casefold_candidates[0]
            status, match_mode = "matched", "casefold_exact"
        elif len(casefold_candidates) > 1:
            candidate = None
            status, match_mode = "ambiguous", "casefold_exact"
        else:
            compact_candidates = distinct_page_candidates(
                by_compact.get(compact_title(title), [])
            )
            if len(compact_candidates) == 1:
                candidate = compact_candidates[0]
                status, match_mode = "matched", "compact_exact"
            elif len(compact_candidates) > 1:
                candidate = None
                status, match_mode = "ambiguous", "compact_exact"
            else:
                candidate = None
                status, match_mode = "missing", None

        if len(casefold_candidates) > 1:
            candidate_page_ids = [value["page_id"] for value in casefold_candidates]
        elif status == "ambiguous":
            candidate_page_ids = [
                value["page_id"]
                for value in distinct_page_candidates(
                    by_compact.get(compact_title(title), [])
                )
            ]
        else:
            candidate_page_ids = []
        resolutions.append(
            {
                "requested_title": title,
                "casefold_key": casefold_title(title),
                "compact_key": compact_title(title),
                "context_occurrences": title_occurrences[title],
                "status": status,
                "match_mode": match_mode,
                "matched_page_id": candidate["page_id"] if candidate else None,
                "matched_page_title": candidate["title"] if candidate else None,
                "candidate_page_ids": candidate_page_ids,
            }
        )
    return resolutions


def page_plaintext(page: dict[str, Any]) -> str:
    paragraphs = []
    for sentences in page["text"]:
        paragraph = html.unescape(LINK_TAG.sub("", "".join(sentences))).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip() + "\n"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def materialize_selected_pages(
    *,
    candidate_path: Path,
    resolutions: list[dict[str, Any]],
    by_casefold: dict[str, list[dict[str, Any]]],
    by_compact: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    include_ambiguous_exact_pages: bool,
) -> list[dict[str, Any]]:
    selected_page_ids = {
        resolution["matched_page_id"]
        for resolution in resolutions
        if resolution["status"] == "matched"
    }
    if include_ambiguous_exact_pages:
        for resolution in resolutions:
            if (
                resolution["status"] == "ambiguous"
                and resolution["match_mode"] == "casefold_exact"
            ):
                selected_page_ids.update(resolution["candidate_page_ids"])

    candidates_by_id = {}
    for candidate_lists in (by_casefold.values(), by_compact.values()):
        for candidate_list in candidate_lists:
            for candidate in candidate_list:
                if candidate["page_id"] not in selected_page_ids:
                    continue
                previous = candidates_by_id.get(candidate["page_id"])
                if previous is not None and previous != candidate:
                    raise ValueError(
                        f"conflicting candidate for page {candidate['page_id']}"
                    )
                candidates_by_id[candidate["page_id"]] = candidate

    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True)
    raw_documents_path = output_dir / "dataset_info/documents_raw.jsonl"
    raw_documents_path.parent.mkdir(parents=True)
    documents = []
    with (
        candidate_path.open("rb") as candidates,
        raw_documents_path.open("wb") as raw_output,
    ):
        for page_id in sorted(candidates_by_id, key=int):
            candidate = candidates_by_id[page_id]
            candidates.seek(candidate["offset"])
            raw_record = candidates.read(candidate["length"])
            page = json.loads(raw_record)
            if str(page["id"]) != page_id:
                raise ValueError(f"candidate offset mismatch for page {page_id}")
            document_file = f"doc_{page_id}.txt"
            plaintext = page_plaintext(page)
            (documents_dir / document_file).write_text(plaintext, encoding="utf-8")
            raw_output.write(raw_record)
            documents.append(
                {
                    "page_id": page_id,
                    "title": page["title"],
                    "url": page.get("url"),
                    "document_file": f"documents/{document_file}",
                    "character_count": len(plaintext),
                }
            )
    return documents


def build_query_document_records(
    dev_records: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    include_ambiguous_exact_pages: bool,
) -> list[dict[str, Any]]:
    resolution_by_title = {
        resolution["requested_title"]: resolution for resolution in resolutions
    }
    result = []
    for record in dev_records:
        supporting_titles = {
            supporting_fact[0] for supporting_fact in record["supporting_facts"]
        }
        contexts = []
        for title, _ in record["context"]:
            resolution = resolution_by_title[title]
            if (
                include_ambiguous_exact_pages
                and resolution["status"] == "ambiguous"
                and resolution["match_mode"] == "casefold_exact"
            ):
                page_ids = resolution["candidate_page_ids"]
            elif resolution["matched_page_id"] is not None:
                page_ids = [resolution["matched_page_id"]]
            else:
                page_ids = []
            contexts.append(
                {
                    "title": title,
                    "is_supporting_title": title in supporting_titles,
                    "join_status": resolution["status"],
                    "match_mode": resolution["match_mode"],
                    "page_id": resolution["matched_page_id"],
                    "page_ids": page_ids,
                    "matched_page_title": resolution["matched_page_title"],
                }
            )
        result.append({"_id": record["_id"], "contexts": contexts})
    return result


def percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of an empty sequence")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def add_token_statistics(
    *,
    output_dir: Path,
    documents: list[dict[str, Any]],
    tokenizer_name: str,
    chunk_size: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    visible_prefix = ""
    visible_suffix = "\n\n"
    visible_token_overhead = len(
        tokenizer.encode(visible_prefix + visible_suffix, add_special_tokens=False)
    )
    content_chunk_size = chunk_size - visible_token_overhead
    if content_chunk_size <= 0:
        raise ValueError("chunk size does not leave room for document content")

    token_lengths = []
    chunk_counts = []
    for document in documents:
        text = (output_dir / document["document_file"]).read_text(encoding="utf-8")
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        chunk_count = max(1, math.ceil(token_count / content_chunk_size))
        document["token_count"] = token_count
        document["chunk_count"] = chunk_count
        token_lengths.append(token_count)
        chunk_counts.append(chunk_count)

    def distribution(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "min": min(ordered),
            "p50": percentile(ordered, 0.50),
            "p90": percentile(ordered, 0.90),
            "p95": percentile(ordered, 0.95),
            "max": max(ordered),
            "mean": sum(ordered) / len(ordered),
        }

    return {
        "tokenizer": tokenizer_name,
        "add_special_tokens": False,
        "chunk_size": chunk_size,
        "content_chunk_size": content_chunk_size,
        "visible_prefix": visible_prefix,
        "visible_suffix": visible_suffix,
        "visible_token_overhead": visible_token_overhead,
        "chunk_overlap": 0,
        "document_count": len(documents),
        "total_token_count": sum(token_lengths),
        "total_chunk_count": sum(chunk_counts),
        "documents_over_one_chunk": sum(value > 1 for value in chunk_counts),
        "token_count": distribution(token_lengths),
        "chunks_per_document": distribution(chunk_counts),
    }


def validate_official_sources(dev_path: Path, wiki_path: Path) -> dict[str, Any]:
    dev_size = dev_path.stat().st_size
    wiki_size = wiki_path.stat().st_size
    dev_sha256 = file_digest(dev_path, "sha256")
    wiki_md5 = file_digest(wiki_path, "md5")
    if dev_size != EXPECTED_DEV_SIZE or dev_sha256 != EXPECTED_DEV_SHA256:
        raise ValueError(
            "official HotpotQA dev source mismatch: "
            f"bytes={dev_size}, sha256={dev_sha256}"
        )
    if wiki_size != EXPECTED_WIKI_SIZE or wiki_md5 != EXPECTED_WIKI_MD5:
        raise ValueError(
            "official HotpotQA Wikipedia source mismatch: "
            f"bytes={wiki_size}, md5={wiki_md5}"
        )
    return {
        "dev": {
            "url": OFFICIAL_DEV_URL,
            "bytes": dev_size,
            "sha256": dev_sha256,
        },
        "wikipedia": {
            "url": OFFICIAL_WIKI_URL,
            "bytes": wiki_size,
            "md5": wiki_md5,
        },
    }


def prepare_dataset(
    *,
    dev_path: Path,
    wiki_path: Path,
    output_dir: Path,
    tokenizer_name: str,
    chunk_size: int,
    scan_workers: int,
    include_ambiguous_exact_pages: bool,
    verify_official_sources: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_info = (
        validate_official_sources(dev_path, wiki_path)
        if verify_official_sources
        else {
            "dev": {"path": str(dev_path), "bytes": dev_path.stat().st_size},
            "wikipedia": {"path": str(wiki_path), "bytes": wiki_path.stat().st_size},
        }
    )
    dev_records = load_dev_records(dev_path)
    requested_titles, title_occurrences = collect_requested_titles(dev_records)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_directory:
        staging_dir = Path(temporary_directory) / "dataset"
        staging_dir.mkdir()
        candidate_path = Path(temporary_directory) / "candidate_pages.jsonl"
        scan = scan_candidate_pages(
            archive_path=wiki_path,
            requested_titles=requested_titles,
            candidate_path=candidate_path,
            scan_workers=scan_workers,
        )
        resolutions = resolve_titles(
            requested_titles=requested_titles,
            title_occurrences=title_occurrences,
            by_casefold=scan["by_casefold"],
            by_compact=scan["by_compact"],
        )
        documents = materialize_selected_pages(
            candidate_path=candidate_path,
            resolutions=resolutions,
            by_casefold=scan["by_casefold"],
            by_compact=scan["by_compact"],
            output_dir=staging_dir,
            include_ambiguous_exact_pages=include_ambiguous_exact_pages,
        )
        token_statistics = add_token_statistics(
            output_dir=staging_dir,
            documents=documents,
            tokenizer_name=tokenizer_name,
            chunk_size=chunk_size,
        )
        shutil.copy2(dev_path, staging_dir / "hotpot_dev_distractor_v1.json")
        dataset_info = staging_dir / "dataset_info"
        write_jsonl(dataset_info / "title_join.jsonl", resolutions)
        write_jsonl(
            dataset_info / "query_documents.jsonl",
            build_query_document_records(
                dev_records, resolutions, include_ambiguous_exact_pages
            ),
        )
        write_jsonl(dataset_info / "documents.jsonl", documents)

        status_counts = Counter(value["status"] for value in resolutions)
        match_mode_counts = Counter(
            value["match_mode"] for value in resolutions if value["status"] == "matched"
        )
        ambiguous_mode_counts = Counter(
            value["match_mode"]
            for value in resolutions
            if value["status"] == "ambiguous"
        )
        ambiguous_ids = {
            page_id
            for value in resolutions
            if value["status"] == "ambiguous"
            and value["match_mode"] == "casefold_exact"
            for page_id in value["candidate_page_ids"]
        }
        manifest = {
            "dataset": "custom_hotpotqa_distractor_dev_full_wikipedia_documents",
            "description": (
                "Custom corpus formed by exact-normalized title joins from every "
                "HotpotQA distractor-dev context title, including distractors, to "
                "the official processed October 1, 2017 Wikipedia corpus."
            ),
            "sources": source_info,
            "join_policy": {
                "first": "Unicode NFKC plus casefold exact equality",
                "fallback": (
                    "Unicode NFKC plus casefold, retaining only alphanumeric "
                    "characters, followed by exact equality"
                ),
                "fuzzy_matching": False,
                "ambiguous_fallback_matches_are_rejected": True,
                "ambiguous_casefold_exact_pages_in_corpus": include_ambiguous_exact_pages,
            },
            "document_text_policy": {
                "paragraphs": "official sentences concatenated without separators",
                "document": "non-empty paragraphs joined with two newlines",
                "hyperlinks": "only HTML anchor tags removed; entities unescaped",
            },
            "counts": {
                "query_count": len(dev_records),
                "context_occurrence_count": sum(title_occurrences.values()),
                "requested_unique_title_count": len(requested_titles),
                "scanned_wikipedia_page_count": scan["page_count"],
                "candidate_wikipedia_page_count": scan["candidate_count"],
                "extracted_unique_document_count": len(documents),
                "title_join_status": dict(status_counts),
                "title_match_mode": dict(match_mode_counts),
                "ambiguous_title_match_mode": dict(ambiguous_mode_counts),
                "ambiguous_exact_included_unique_document_count": (
                    len(ambiguous_ids) if include_ambiguous_exact_pages else 0
                ),
            },
            "chunk_statistics": token_statistics,
        }
        (dataset_info / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--wiki-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=8,
        help="Processes used to decompress and filter inner Wikipedia shards.",
    )
    parser.add_argument(
        "--include-ambiguous-exact-pages",
        action="store_true",
        help=(
            "Include every page from ambiguous case-insensitive exact-title matches "
            "while retaining the ambiguous join status."
        ),
    )
    parser.add_argument(
        "--skip-official-source-verification",
        action="store_true",
        help="Only for synthetic tests; do not use for the requested dataset build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(
        dev_path=args.dev_path,
        wiki_path=args.wiki_archive,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        scan_workers=args.scan_workers,
        include_ambiguous_exact_pages=args.include_ambiguous_exact_pages,
        verify_official_sources=not args.skip_official_source_verification,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
