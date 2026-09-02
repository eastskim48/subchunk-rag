#!/usr/bin/env python
"""Download and prepare the custom HotpotQA full-Wikipedia RAG dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.hotpotqa.build_rag_inputs import build_rag_inputs, file_sha256
from dataset.hotpotqa.prepare_documents import (
    DEFAULT_TOKENIZER,
    EXPECTED_DEV_SHA256,
    EXPECTED_DEV_SIZE,
    EXPECTED_WIKI_MD5,
    EXPECTED_WIKI_SIZE,
    OFFICIAL_DEV_URL,
    OFFICIAL_WIKI_URL,
    file_digest,
    prepare_dataset,
)
from dataset.hotpotqa.validate_supporting_facts import validate_dataset

EXPECTED_DOCUMENT_COUNT = 66_705


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    filename: str
    size: int
    digest_algorithm: str
    digest: str


DEV_SOURCE = SourceSpec(
    name="HotpotQA distractor development split",
    url=OFFICIAL_DEV_URL,
    filename="hotpot_dev_distractor_v1.json",
    size=EXPECTED_DEV_SIZE,
    digest_algorithm="sha256",
    digest=EXPECTED_DEV_SHA256,
)
WIKI_SOURCE = SourceSpec(
    name="HotpotQA processed Wikipedia archive",
    url=OFFICIAL_WIKI_URL,
    filename="enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2",
    size=EXPECTED_WIKI_SIZE,
    digest_algorithm="md5",
    digest=EXPECTED_WIKI_MD5,
)


def log(message: str) -> None:
    print(f"[hotpotqa-prepare] {message}", flush=True)


def verify_source(path: Path, spec: SourceSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec.name}: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ValueError(
            f"{spec.name} size mismatch: {actual_size}, expected {spec.size}: {path}"
        )
    actual_digest = file_digest(path, spec.digest_algorithm)
    if actual_digest != spec.digest:
        raise ValueError(
            f"{spec.name} {spec.digest_algorithm} mismatch: "
            f"{actual_digest}, expected {spec.digest}: {path}"
        )


def _quarantine_partial(path: Path) -> Path:
    suffix = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantine = path.with_name(f"{path.name}.invalid-{suffix}")
    counter = 1
    while quarantine.exists():
        quarantine = path.with_name(f"{path.name}.invalid-{suffix}-{counter}")
        counter += 1
    path.rename(quarantine)
    return quarantine


def download_source(
    spec: SourceSpec,
    destination: Path,
    *,
    timeout_sec: float = 60.0,
) -> Path:
    """Download one official source with resume and exact verification."""

    if destination.exists():
        verify_source(destination, spec)
        log(f"verified existing source: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > spec.size:
        quarantine = _quarantine_partial(partial)
        raise ValueError(
            f"partial download exceeds expected size; retained as {quarantine}"
        )

    headers = {"User-Agent": "subchunk-dataset-preparer/1"}
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
        log(f"resuming {spec.name} at byte {partial_size:,}")
    else:
        log(f"downloading {spec.name} from {spec.url}")

    request = Request(spec.url, headers=headers)
    with urlopen(request, timeout=timeout_sec) as response:
        status = getattr(response, "status", None)
        append = partial_size > 0 and status == 206
        if partial_size and not append:
            log("source server did not accept resume; restarting the partial download")
        mode = "ab" if append else "wb"
        downloaded = partial_size if append else 0
        next_report = downloaded + 256 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    log(f"downloaded {downloaded:,} / {spec.size:,} bytes")
                    next_report = downloaded + 256 * 1024 * 1024
            output.flush()
            os.fsync(output.fileno())

    try:
        verify_source(partial, spec)
    except (OSError, ValueError):
        quarantine = _quarantine_partial(partial)
        raise ValueError(
            f"downloaded {spec.name} failed verification; retained as {quarantine}"
        ) from None
    partial.replace(destination)
    log(f"download complete and verified: {destination}")
    return destination


def _required_state(paths: list[Path], stage: str) -> bool:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return False
    if len(existing) != len(paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise RuntimeError(
            f"{stage} is incomplete; refusing to overwrite it. Missing: {missing}"
        )
    return True


def verify_corpus(dataset_root: Path) -> dict[str, Any]:
    dataset_info = dataset_root / "dataset_info"
    required = [
        dataset_root / "hotpot_dev_distractor_v1.json",
        dataset_root / "documents",
        dataset_info / "manifest.json",
        dataset_info / "documents.jsonl",
        dataset_info / "documents_raw.jsonl",
        dataset_info / "query_documents.jsonl",
        dataset_info / "title_join.jsonl",
    ]
    if not _required_state(required, "document-corpus stage"):
        raise RuntimeError(f"missing document corpus: {dataset_root}")

    manifest = json.loads(required[2].read_text(encoding="utf-8"))
    if manifest.get("dataset") != (
        "custom_hotpotqa_distractor_dev_full_wikipedia_documents"
    ):
        raise ValueError(f"unexpected dataset manifest: {required[2]}")
    expected = manifest.get("counts", {}).get("extracted_unique_document_count")
    if expected != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            f"manifest document count is {expected}, expected {EXPECTED_DOCUMENT_COUNT}"
        )
    actual = sum(path.is_file() for path in required[1].iterdir())
    if actual != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            f"materialized document count is {actual}, expected {EXPECTED_DOCUMENT_COUNT}"
        )
    return manifest


def ensure_supporting_fact_validation(dataset_root: Path) -> dict[str, Any]:
    dataset_info = dataset_root / "dataset_info"
    required = [
        dataset_info / "title_to_documents.json",
        dataset_info / "document_file_to_title.json",
        dataset_info / "supporting_fact_validation.jsonl",
        dataset_info / "supporting_fact_validation_summary.json",
    ]
    if _required_state(required, "supporting-fact validation stage"):
        summary = json.loads(required[3].read_text(encoding="utf-8"))
        if summary.get("supporting_fact_count") != 18_005:
            raise ValueError(f"unexpected supporting-fact summary: {required[3]}")
        log("verified existing supporting-fact validation")
        return summary

    log("validating supporting facts")
    summary = validate_dataset(dataset_root)
    log("supporting-fact validation complete")
    return summary


def ensure_rag_inputs(dataset_root: Path) -> dict[str, Any]:
    required = [
        dataset_root / "questions/query.jsonl",
        dataset_root / "answers/answer.jsonl",
        dataset_root / "dataset_info/evidence_labels.jsonl",
        dataset_root / "dataset_info/rag_input_validation_manifest.json",
    ]
    if _required_state(required, "RAG-input stage"):
        manifest = json.loads(required[3].read_text(encoding="utf-8"))
        expected_hashes = manifest.get("outputs", {})
        for path in required[:3]:
            relative = str(path.relative_to(dataset_root))
            expected = expected_hashes.get(relative)
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"RAG input hash mismatch for {path}: {actual}, expected {expected}"
                )
        log("verified existing query, answer, and evidence inputs")
        return manifest

    log("building query, answer, and evidence inputs")
    manifest = build_rag_inputs(dataset_root)
    log("query, answer, and evidence input construction complete")
    return manifest


def prepare_hotpotqa(
    *,
    output_dir: Path,
    source_dir: Path | None,
    dev_path: Path | None,
    wiki_archive: Path | None,
    tokenizer_name: str,
    chunk_size: int,
    scan_workers: int,
    download_timeout_sec: float,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        log(f"verifying existing document corpus: {output_dir}")
        corpus_manifest = verify_corpus(output_dir)
    else:
        resolved_source_dir = (
            source_dir.resolve()
            if source_dir is not None
            else output_dir.parent / f"{output_dir.name}-sources"
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
        log(f"building document corpus in {output_dir}")
        corpus_manifest = prepare_dataset(
            dev_path=resolved_dev,
            wiki_path=resolved_wiki,
            output_dir=output_dir,
            tokenizer_name=tokenizer_name,
            chunk_size=chunk_size,
            scan_workers=scan_workers,
            include_ambiguous_exact_pages=True,
            verify_official_sources=True,
        )
        verify_corpus(output_dir)
        log("document corpus construction complete")

    validation_summary = ensure_supporting_fact_validation(output_dir)
    rag_manifest = ensure_rag_inputs(output_dir)
    return {
        "dataset_root": str(output_dir),
        "dataset": corpus_manifest["dataset"],
        "document_count": corpus_manifest["counts"]["extracted_unique_document_count"],
        "query_count": rag_manifest["counts"]["query_count"],
        "supporting_fact_count": validation_summary["supporting_fact_count"],
        "outputs": {
            "documents": str(output_dir / "documents"),
            "queries": str(output_dir / "questions/query.jsonl"),
            "answers": str(output_dir / "answers/answer.jsonl"),
            "evidence": str(output_dir / "dataset_info/evidence_labels.jsonl"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Download directory. Defaults to <output-dir-name>-sources beside "
            "the output directory."
        ),
    )
    parser.add_argument("--dev-path", type=Path)
    parser.add_argument("--wiki-archive", type=Path)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--download-timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_hotpotqa(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        dev_path=args.dev_path,
        wiki_archive=args.wiki_archive,
        tokenizer_name=args.tokenizer,
        chunk_size=args.chunk_size,
        scan_workers=args.scan_workers,
        download_timeout_sec=args.download_timeout_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    dataset_root = shlex.quote(str(args.output_dir.resolve()))
    log("dataset inputs are complete; next build the retrieval DB and ColBERT artifact")
    log(
        f"set DATASET to this dataset root for the existing build scripts: {dataset_root}"
    )


if __name__ == "__main__":
    main()
