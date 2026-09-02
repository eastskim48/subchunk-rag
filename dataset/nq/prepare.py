#!/usr/bin/env python
"""Download and prepare the custom DAPR-NQ/NQ-open RAG dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.nq.build import (
    DAPR_CONFIG,
    DAPR_DATASET,
    DAPR_SNAPSHOT_REVISION,
    EXPECTED_DAPR_DOCUMENTS,
    EXPECTED_DAPR_PASSAGES,
    EXPECTED_INTERSECTION_QRELS,
    EXPECTED_INTERSECTION_QUERIES,
    NQ_OPEN_DEV_URL,
    build_dataset,
    file_sha256,
    load_jsonl,
    validate_output,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    relative_path: str
    size: int
    sha256: str


def dapr_url(relative_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DAPR_DATASET}/resolve/"
        f"{DAPR_SNAPSHOT_REVISION}/{DAPR_CONFIG}/{relative_path}?download=true"
    )


SOURCES = (
    SourceSpec(
        name="DAPR-NQ document shard 0",
        url=dapr_url("docs/test-00000-of-00003.parquet"),
        relative_path="NaturalQuestions/docs/test-00000-of-00003.parquet",
        size=589_507_087,
        sha256="8050f5652d887484700680ffe3b65df716b607362e1a54c01323a622e34e4f98",
    ),
    SourceSpec(
        name="DAPR-NQ document shard 1",
        url=dapr_url("docs/test-00001-of-00003.parquet"),
        relative_path="NaturalQuestions/docs/test-00001-of-00003.parquet",
        size=148_850_074,
        sha256="373b6027a75e143ca3abe2a48c4a0e61cc3304a32e50ee192dec409bdb187556",
    ),
    SourceSpec(
        name="DAPR-NQ document shard 2",
        url=dapr_url("docs/test-00002-of-00003.parquet"),
        relative_path="NaturalQuestions/docs/test-00002-of-00003.parquet",
        size=38_967_544,
        sha256="fb598929e09a066694bbf7424dd943e216d90ff44f66ca5197c9195232c44f46",
    ),
    SourceSpec(
        name="DAPR-NQ test queries",
        url=dapr_url("queries/test.parquet"),
        relative_path="NaturalQuestions/queries/test.parquet",
        size=147_008,
        sha256="1c6043901f136728a7a868ed81f0f84f278950aaa474d4aba8792511467d7355",
    ),
    SourceSpec(
        name="DAPR-NQ test qrels",
        url=dapr_url("qrels/test.parquet"),
        relative_path="NaturalQuestions/qrels/test.parquet",
        size=56_456,
        sha256="8cbff4836dcd9e155a22d6d8127a3c49bef67ec5755567649f0b4ea3fe1fccb2",
    ),
    SourceSpec(
        name="official NQ-open development split",
        url=NQ_OPEN_DEV_URL,
        relative_path="NQ-open.dev.jsonl",
        size=391_316,
        sha256="f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616",
    ),
)


def log(message: str) -> None:
    print(f"[nq-prepare] {message}", flush=True)


def verify_source(path: Path, spec: SourceSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec.name}: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ValueError(
            f"{spec.name} size mismatch: {actual_size}, expected {spec.size}: {path}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"{spec.name} SHA-256 mismatch: {actual_sha256}, "
            f"expected {spec.sha256}: {path}"
        )


def quarantine_partial(path: Path) -> Path:
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
    timeout_sec: float,
) -> Path:
    if destination.exists():
        verify_source(destination, spec)
        log(f"verified existing source: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > spec.size:
        quarantine = quarantine_partial(partial)
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
        quarantine = quarantine_partial(partial)
        raise ValueError(
            f"downloaded {spec.name} failed verification; retained as {quarantine}"
        ) from None
    partial.replace(destination)
    log(f"download complete and verified: {destination}")
    return destination


def verify_existing_dataset(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "dataset_info/manifest.json"
    questions_path = output_dir / "questions/query.jsonl"
    answers_path = output_dir / "answers/answer.jsonl"
    evidence_path = output_dir / "dataset_info/evidence_labels.jsonl"
    documents_dir = output_dir / "documents"
    required = [
        manifest_path,
        questions_path,
        answers_path,
        evidence_path,
        documents_dir,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"existing dataset is incomplete; refusing to overwrite it. Missing: {missing}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dapr_nq_nq_open_intersection_rag":
        raise ValueError(f"unexpected dataset manifest: {manifest_path}")
    expected_counts = {
        "intersection_queries": EXPECTED_INTERSECTION_QUERIES,
        "intersection_qrels": EXPECTED_INTERSECTION_QRELS,
        "documents": EXPECTED_DAPR_DOCUMENTS,
        "passages": EXPECTED_DAPR_PASSAGES,
    }
    counts = manifest.get("counts", {})
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise ValueError(
                f"manifest count {key} is {counts.get(key)}, expected {expected}"
            )

    questions = load_jsonl(questions_path)
    answers = load_jsonl(answers_path)
    evidence = load_jsonl(evidence_path)
    if not (
        len(questions) == len(answers) == len(evidence) == EXPECTED_INTERSECTION_QUERIES
    ):
        raise ValueError("query, answer, or evidence record count mismatch")
    actual_documents = sum(path.is_file() for path in documents_dir.iterdir())
    if actual_documents != EXPECTED_DAPR_DOCUMENTS:
        raise ValueError(
            f"document count is {actual_documents}, expected {EXPECTED_DAPR_DOCUMENTS}"
        )
    validate_output(
        staging_dir=output_dir,
        questions=questions,
        answers=answers,
        evidence_records=evidence,
    )
    return manifest


def prepare_nq(
    *,
    output_dir: Path,
    source_dir: Path | None,
    download_timeout_sec: float,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        log(f"verifying existing dataset: {output_dir}")
        manifest = verify_existing_dataset(output_dir)
        log("existing dataset verification complete")
        return manifest

    resolved_source_dir = (
        source_dir.resolve()
        if source_dir is not None
        else output_dir.parent / f"{output_dir.name}-sources"
    )
    downloaded = {
        spec.relative_path: download_source(
            spec,
            resolved_source_dir / spec.relative_path,
            timeout_sec=download_timeout_sec,
        )
        for spec in SOURCES
    }
    log(f"building custom dataset in {output_dir}")
    manifest = build_dataset(
        dapr_root=resolved_source_dir / "NaturalQuestions",
        nq_open_dev_file=downloaded["NQ-open.dev.jsonl"],
        output_dir=output_dir,
    )
    verify_existing_dataset(output_dir)
    log("dataset construction and verification complete")
    return manifest


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
    parser.add_argument("--download-timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_nq(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        download_timeout_sec=args.download_timeout_sec,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
