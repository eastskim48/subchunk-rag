#!/usr/bin/env python
"""Prepare HotpotQA queries against every page in the official Wikipedia archive."""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
from collections import Counter, defaultdict
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.hotpotqa.prepare import (
    DEV_SOURCE,
    WIKI_SOURCE,
    download_source,
    ensure_rag_inputs,
    log,
    verify_source,
)
from dataset.hotpotqa.prepare_documents import (
    build_query_document_records,
    casefold_title,
    collect_requested_titles,
    compact_title,
    load_dev_records,
    page_plaintext,
    resolve_titles,
    validate_official_sources,
    validate_wikipedia_page,
    write_jsonl,
)
from dataset.hotpotqa.validate_supporting_facts import (
    validate_dataset,
    write_json_atomic,
)

EXPECTED_FULL_PAGE_COUNT = 5_486_212
FULL_DATASET_NAME = "custom_hotpotqa_distractor_dev_complete_processed_wikipedia"

_REQUESTED_CASEFOLD: set[str] = set()
_REQUESTED_COMPACT: set[str] = set()


def _initialize_full_worker(
    requested_casefold: set[str], requested_compact: set[str]
) -> None:
    global _REQUESTED_CASEFOLD, _REQUESTED_COMPACT
    _REQUESTED_CASEFOLD = requested_casefold
    _REQUESTED_COMPACT = requested_compact


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def log_storage_capacity(path: Path) -> None:
    usage = shutil.disk_usage(path)
    filesystem = os.statvfs(path)
    log(
        "full-corpus target capacity: "
        f"free_bytes={usage.free:,}, free_inodes={filesystem.f_favail:,}, "
        f"expected_document_files={EXPECTED_FULL_PAGE_COUNT:,}"
    )


def _materialize_full_shard(
    *,
    shard_index: int,
    member_name: str,
    compressed_shard: bytes,
    documents_dir: str,
    shard_metadata_dir: str,
) -> dict[str, Any]:
    documents_path = Path(documents_dir)
    metadata_path = Path(shard_metadata_dir)
    prefix = f"{shard_index:05d}"
    document_records = metadata_path / f"{prefix}.documents.jsonl"
    candidate_pages = metadata_path / f"{prefix}.context_candidates.jsonl"
    document_records_part = document_records.with_suffix(".jsonl.part")
    candidate_pages_part = candidate_pages.with_suffix(".jsonl.part")
    page_count = 0
    candidate_count = 0

    with (
        bz2.open(
            io.BytesIO(compressed_shard),
            mode="rt",
            encoding="utf-8",
        ) as shard,
        document_records_part.open("w", encoding="utf-8") as metadata_output,
        candidate_pages_part.open("w", encoding="utf-8") as candidate_output,
    ):
        for line_number, line in enumerate(shard, start=1):
            if not line.strip():
                continue
            page = json.loads(line)
            if not isinstance(page, dict):
                raise ValueError(f"{member_name}:{line_number} is not a JSON object")
            validate_wikipedia_page(page)
            page_id = str(page["id"])
            if re.fullmatch(r"[0-9]+", page_id) is None:
                raise ValueError(
                    f"{member_name}:{line_number} has a non-decimal page ID: {page_id!r}"
                )
            document_file = f"doc_{page_id}.txt"
            text = page_plaintext(page)
            (documents_path / document_file).write_text(text, encoding="utf-8")
            metadata_output.write(
                _json_line(
                    {
                        "page_id": page_id,
                        "title": page["title"],
                        "url": page.get("url"),
                        "document_file": f"documents/{document_file}",
                        "character_count": len(text),
                    }
                )
            )
            page_count += 1
            if (
                casefold_title(page["title"]) in _REQUESTED_CASEFOLD
                or compact_title(page["title"]) in _REQUESTED_COMPACT
            ):
                candidate_output.write(_json_line(page))
                candidate_count += 1

        for output in (metadata_output, candidate_output):
            output.flush()
            os.fsync(output.fileno())

    document_records_part.replace(document_records)
    candidate_pages_part.replace(candidate_pages)
    summary = {
        "shard_index": shard_index,
        "member_name": member_name,
        "page_count": page_count,
        "context_candidate_page_count": candidate_count,
        "document_records": document_records.name,
        "context_candidates": candidate_pages.name,
    }
    write_json_atomic(metadata_path / f"{prefix}.summary.json", summary)
    return summary


def _load_completed_shard(
    summary_path: Path, *, shard_index: int, member_name: str
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("shard_index") != shard_index
        or summary.get("member_name") != member_name
    ):
        raise ValueError(f"incompatible shard checkpoint: {summary_path}")
    for key in ("document_records", "context_candidates"):
        if not (summary_path.parent / summary[key]).is_file():
            raise FileNotFoundError(
                f"shard checkpoint references missing {key}: {summary_path}"
            )
    return summary


def materialize_all_wikipedia_pages(
    *,
    archive_path: Path,
    staging_dir: Path,
    requested_titles: list[str],
    scan_workers: int,
) -> list[dict[str, Any]]:
    if scan_workers <= 0:
        raise ValueError("scan_workers must be positive")
    documents_dir = staging_dir / "documents"
    shard_metadata_dir = staging_dir / "dataset_info/full_wikipedia_shards"
    documents_dir.mkdir(parents=True, exist_ok=True)
    shard_metadata_dir.mkdir(parents=True, exist_ok=True)
    requested_casefold = {casefold_title(title) for title in requested_titles}
    requested_compact = {compact_title(title) for title in requested_titles}
    summaries: list[dict[str, Any]] = []

    def completed_or_none(shard_index: int, member_name: str):
        summary_path = shard_metadata_dir / f"{shard_index:05d}.summary.json"
        if not summary_path.is_file():
            return None
        summary = _load_completed_shard(
            summary_path, shard_index=shard_index, member_name=member_name
        )
        log(
            f"verified completed Wikipedia shard {shard_index}: "
            f"{summary['page_count']:,} pages"
        )
        return summary

    with tarfile.open(archive_path, mode="r:bz2") as archive:
        if scan_workers == 1:
            _initialize_full_worker(requested_casefold, requested_compact)
            shard_index = 0
            for member in archive:
                if not member.isfile():
                    continue
                completed = completed_or_none(shard_index, member.name)
                if completed is not None:
                    summaries.append(completed)
                    shard_index += 1
                    continue
                member_handle = archive.extractfile(member)
                if member_handle is None:
                    raise ValueError(f"cannot read Wikipedia shard: {member.name}")
                summary = _materialize_full_shard(
                    shard_index=shard_index,
                    member_name=member.name,
                    compressed_shard=member_handle.read(),
                    documents_dir=str(documents_dir),
                    shard_metadata_dir=str(shard_metadata_dir),
                )
                summaries.append(summary)
                log(
                    f"completed Wikipedia shard {shard_index}: "
                    f"{summary['page_count']:,} pages"
                )
                shard_index += 1
        else:
            max_pending = max(1, scan_workers * 2)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=scan_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_full_worker,
                initargs=(requested_casefold, requested_compact),
            ) as executor:
                pending: list[tuple[int, concurrent.futures.Future]] = []
                completed_by_index: dict[int, dict[str, Any]] = {}
                shard_index = 0

                def consume_one() -> None:
                    index, future = pending.pop(0)
                    summary = future.result()
                    completed_by_index[index] = summary
                    log(
                        f"completed Wikipedia shard {index}: "
                        f"{summary['page_count']:,} pages"
                    )

                for member in archive:
                    if not member.isfile():
                        continue
                    completed = completed_or_none(shard_index, member.name)
                    if completed is not None:
                        completed_by_index[shard_index] = completed
                    else:
                        member_handle = archive.extractfile(member)
                        if member_handle is None:
                            raise ValueError(
                                f"cannot read Wikipedia shard: {member.name}"
                            )
                        pending.append(
                            (
                                shard_index,
                                executor.submit(
                                    _materialize_full_shard,
                                    shard_index=shard_index,
                                    member_name=member.name,
                                    compressed_shard=member_handle.read(),
                                    documents_dir=str(documents_dir),
                                    shard_metadata_dir=str(shard_metadata_dir),
                                ),
                            )
                        )
                    if len(pending) >= max_pending:
                        consume_one()
                    shard_index += 1
                while pending:
                    consume_one()
                summaries = [completed_by_index[index] for index in range(shard_index)]

    return summaries


def _write_combined_jsonl(output_path: Path, inputs: list[Path]) -> None:
    temporary = output_path.with_name(f".{output_path.name}.part")
    with temporary.open("wb") as output:
        for input_path in inputs:
            with input_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(output_path)


def finalize_full_corpus_metadata(
    *,
    staging_dir: Path,
    dev_records: list[dict[str, Any]],
    requested_titles: list[str],
    title_occurrences: Counter[str],
    shard_summaries: list[dict[str, Any]],
    source_info: dict[str, Any],
) -> dict[str, Any]:
    dataset_info = staging_dir / "dataset_info"
    shard_dir = dataset_info / "full_wikipedia_shards"
    _write_combined_jsonl(
        dataset_info / "documents.jsonl",
        [shard_dir / summary["document_records"] for summary in shard_summaries],
    )
    candidate_paths = [
        shard_dir / summary["context_candidates"] for summary in shard_summaries
    ]
    _write_combined_jsonl(dataset_info / "documents_raw.jsonl", candidate_paths)

    by_casefold: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_compact: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    context_documents_by_id: dict[str, dict[str, Any]] = {}
    for candidate_path in candidate_paths:
        with candidate_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                page = json.loads(line)
                page_id = str(page["id"])
                candidate = {"page_id": page_id, "title": page["title"]}
                by_casefold[casefold_title(page["title"])].append(candidate)
                by_compact[compact_title(page["title"])].append(candidate)
                text = page_plaintext(page)
                context_documents_by_id[page_id] = {
                    "page_id": page_id,
                    "title": page["title"],
                    "url": page.get("url"),
                    "document_file": f"documents/doc_{page_id}.txt",
                    "character_count": len(text),
                }

    resolutions = resolve_titles(
        requested_titles=requested_titles,
        title_occurrences=title_occurrences,
        by_casefold=dict(by_casefold),
        by_compact=dict(by_compact),
    )
    context_documents = sorted(
        context_documents_by_id.values(), key=lambda value: int(value["page_id"])
    )
    write_jsonl(dataset_info / "context_documents.jsonl", context_documents)
    write_jsonl(dataset_info / "title_join.jsonl", resolutions)
    write_jsonl(
        dataset_info / "query_documents.jsonl",
        build_query_document_records(
            dev_records, resolutions, include_ambiguous_exact_pages=True
        ),
    )

    status_counts = Counter(value["status"] for value in resolutions)
    match_mode_counts = Counter(
        value["match_mode"] for value in resolutions if value["status"] == "matched"
    )
    page_count = sum(summary["page_count"] for summary in shard_summaries)
    manifest = {
        "dataset": FULL_DATASET_NAME,
        "description": (
            "Custom HotpotQA distractor-development queries evaluated against "
            "every page in the official processed October 1, 2017 Wikipedia archive."
        ),
        "custom_protocol": True,
        "sources": source_info,
        "retrieval_corpus_policy": {
            "scope": "all_pages_in_official_processed_wikipedia_archive",
            "query_specific_context_filtering": False,
            "pooled_dev_context_title_filtering": False,
        },
        "join_policy": {
            "first": "Unicode NFKC plus casefold exact equality",
            "fallback": (
                "Unicode NFKC plus casefold, retaining only alphanumeric "
                "characters, followed by exact equality"
            ),
            "fuzzy_matching": False,
            "ambiguous_fallback_matches_are_rejected": True,
            "ambiguous_casefold_exact_pages_in_corpus": True,
        },
        "document_text_policy": {
            "paragraphs": "official sentences concatenated without separators",
            "document": "non-empty paragraphs joined with two newlines",
            "hyperlinks": "only HTML anchor tags removed; entities unescaped",
        },
        "metadata_policy": {
            "documents": "dataset_info/documents.jsonl contains every page",
            "context_documents": (
                "dataset_info/context_documents.jsonl contains only pages needed "
                "for supporting-fact title validation"
            ),
            "raw_pages": (
                "dataset_info/documents_raw.jsonl retains raw pages whose titles "
                "match any development-context title under an exact join key"
            ),
        },
        "counts": {
            "query_count": len(dev_records),
            "requested_unique_title_count": len(requested_titles),
            "scanned_wikipedia_page_count": page_count,
            "extracted_unique_document_count": page_count,
            "context_candidate_page_count": len(context_documents),
            "title_join_status": dict(status_counts),
            "title_match_mode": dict(match_mode_counts),
            "archive_shard_count": len(shard_summaries),
        },
    }
    write_json_atomic(dataset_info / "manifest.json", manifest)
    return manifest


def prepare_full_corpus(
    *,
    dev_path: Path,
    wiki_archive: Path,
    output_dir: Path,
    scan_workers: int,
    expected_page_count: int = EXPECTED_FULL_PAGE_COUNT,
    verify_official_sources: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_info = (
        validate_official_sources(dev_path, wiki_archive)
        if verify_official_sources
        else {
            "dev": {"path": str(dev_path), "bytes": dev_path.stat().st_size},
            "wikipedia": {
                "path": str(wiki_archive),
                "bytes": wiki_archive.stat().st_size,
            },
        }
    )
    dev_records = load_dev_records(dev_path)
    requested_titles, title_occurrences = collect_requested_titles(dev_records)
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    dataset_info = staging_dir / "dataset_info"
    state_path = dataset_info / "full_corpus_build_state.json"
    expected_state = {
        "dataset": FULL_DATASET_NAME,
        "dev_source": source_info["dev"],
        "wikipedia_source": source_info["wikipedia"],
        "expected_page_count": expected_page_count,
    }
    if staging_dir.exists():
        if not state_path.is_file():
            raise RuntimeError(
                f"full-corpus staging directory has no build state: {staging_dir}"
            )
        actual_state = json.loads(state_path.read_text(encoding="utf-8"))
        if actual_state != expected_state:
            raise ValueError(f"incompatible full-corpus build state: {state_path}")
        log(f"resuming full-Wikipedia corpus staging directory: {staging_dir}")
    else:
        dataset_info.mkdir(parents=True)
        write_json_atomic(state_path, expected_state)

    shard_summaries = materialize_all_wikipedia_pages(
        archive_path=wiki_archive,
        staging_dir=staging_dir,
        requested_titles=requested_titles,
        scan_workers=scan_workers,
    )
    actual_page_count = sum(summary["page_count"] for summary in shard_summaries)
    if actual_page_count != expected_page_count:
        raise ValueError(
            f"Wikipedia page count is {actual_page_count}, expected {expected_page_count}"
        )
    shutil.copy2(dev_path, staging_dir / "hotpot_dev_distractor_v1.json")
    manifest = finalize_full_corpus_metadata(
        staging_dir=staging_dir,
        dev_records=dev_records,
        requested_titles=requested_titles,
        title_occurrences=title_occurrences,
        shard_summaries=shard_summaries,
        source_info=source_info,
    )
    staging_dir.rename(output_dir)
    return manifest


def verify_full_corpus(
    dataset_root: Path,
    *,
    expected_page_count: int = EXPECTED_FULL_PAGE_COUNT,
) -> dict[str, Any]:
    required = [
        dataset_root / "documents",
        dataset_root / "hotpot_dev_distractor_v1.json",
        dataset_root / "dataset_info/manifest.json",
        dataset_root / "dataset_info/documents.jsonl",
        dataset_root / "dataset_info/context_documents.jsonl",
        dataset_root / "dataset_info/documents_raw.jsonl",
        dataset_root / "dataset_info/title_join.jsonl",
        dataset_root / "dataset_info/query_documents.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete full-Wikipedia corpus; missing: {missing}")
    manifest = json.loads(required[2].read_text(encoding="utf-8"))
    if manifest.get("dataset") != FULL_DATASET_NAME:
        raise ValueError(f"unexpected full-corpus manifest: {required[2]}")
    if manifest.get("retrieval_corpus_policy") != {
        "scope": "all_pages_in_official_processed_wikipedia_archive",
        "query_specific_context_filtering": False,
        "pooled_dev_context_title_filtering": False,
    }:
        raise ValueError(f"unexpected retrieval corpus policy: {required[2]}")
    page_count = manifest.get("counts", {}).get("extracted_unique_document_count")
    if page_count != expected_page_count:
        raise ValueError(
            f"full-corpus manifest has {page_count} pages, "
            f"expected {expected_page_count}: {required[2]}"
        )
    return manifest


def ensure_full_supporting_fact_validation(dataset_root: Path) -> dict[str, Any]:
    summary_path = dataset_root / "dataset_info/supporting_fact_validation_summary.json"
    validation_path = dataset_root / "dataset_info/supporting_fact_validation.jsonl"
    title_index_path = dataset_root / "dataset_info/title_to_documents.json"
    reverse_index_path = dataset_root / "dataset_info/document_file_to_title.json"
    required = [summary_path, validation_path, title_index_path, reverse_index_path]
    existing = [path for path in required if path.exists()]
    if existing and len(existing) != len(required):
        missing = [str(path) for path in required if not path.exists()]
        raise RuntimeError(
            "supporting-fact validation is incomplete; refusing to overwrite it. "
            f"Missing: {missing}"
        )
    if len(existing) == len(required):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("supporting_fact_count") != 18_005:
            raise ValueError(f"unexpected supporting-fact summary: {summary_path}")
        log("verified existing full-corpus supporting-fact validation")
        return summary
    log("validating supporting facts against the full corpus title subset")
    return validate_dataset(
        dataset_root,
        documents_index_path=dataset_root / "dataset_info/context_documents.jsonl",
    )


def prepare_hotpot_full(
    *,
    output_dir: Path,
    source_dir: Path | None,
    dev_path: Path | None,
    wiki_archive: Path | None,
    scan_workers: int,
    download_timeout_sec: float,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        log(f"verifying existing full-Wikipedia corpus: {output_dir}")
        corpus_manifest = verify_full_corpus(output_dir)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_storage_capacity(output_dir.parent)
        resolved_source_dir = (
            source_dir.resolve()
            if source_dir is not None
            else output_dir.parent / "hotpotqa-sources"
        )
        resolved_dev = (
            dev_path.resolve()
            if dev_path is not None
            else download_source(
                DEV_SOURCE,
                resolved_source_dir / DEV_SOURCE.filename,
                timeout_sec=download_timeout_sec,
            )
        )
        resolved_wiki = (
            wiki_archive.resolve()
            if wiki_archive is not None
            else download_source(
                WIKI_SOURCE,
                resolved_source_dir / WIKI_SOURCE.filename,
                timeout_sec=download_timeout_sec,
            )
        )
        verify_source(resolved_dev, DEV_SOURCE)
        verify_source(resolved_wiki, WIKI_SOURCE)
        corpus_manifest = prepare_full_corpus(
            dev_path=resolved_dev,
            wiki_archive=resolved_wiki,
            output_dir=output_dir,
            scan_workers=scan_workers,
        )
        verify_full_corpus(output_dir)

    validation_summary = ensure_full_supporting_fact_validation(output_dir)
    rag_manifest = ensure_rag_inputs(output_dir)
    return {
        "dataset_root": str(output_dir),
        "dataset": corpus_manifest["dataset"],
        "document_count": corpus_manifest["counts"]["extracted_unique_document_count"],
        "query_count": rag_manifest["counts"]["query_count"],
        "supporting_fact_count": validation_summary["supporting_fact_count"],
        "retrieval_corpus_policy": corpus_manifest["retrieval_corpus_policy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--dev-path", type=Path)
    parser.add_argument("--wiki-archive", type=Path)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--download-timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_hotpot_full(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        dev_path=args.dev_path,
        wiki_archive=args.wiki_archive,
        scan_workers=args.scan_workers,
        download_timeout_sec=args.download_timeout_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
