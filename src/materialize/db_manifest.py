"""Build and verify reproducibility metadata for a retrieval database."""

import json
import hashlib
import os
from pathlib import Path
from typing import Any

DB_BUILD_MANIFEST_FILENAME = "build_manifest.json"
DB_BUILD_MANIFEST_FORMAT = "subchunk_db_build_manifest"
DB_BUILD_MANIFEST_VERSION = 1


def normalize_embedding_backend(embedding_backend: str) -> str:
    normalized = embedding_backend.strip().lower()
    if normalized == "chroma_default":
        return "default"
    return normalized


def build_db_manifest(
    *,
    splitter: str,
    merger: str | None,
    cacheable_chunk_size: int | None,
    retrievable_chunk_size: int | None,
    max_subchunk_tokens: int | None,
    tokenizer_name: str,
    dummy_bos_count: int,
    sentence_cache_token_format: str,
    deduplicate_documents_by_hash: bool,
    embedding_backend: str,
    db_batch_size: int,
    embedding_device: str,
    embedding_batch_size: int,
) -> dict[str, Any]:
    return {
        "format": DB_BUILD_MANIFEST_FORMAT,
        "format_version": DB_BUILD_MANIFEST_VERSION,
        "splitter": splitter,
        "merger": merger,
        "cacheable_chunk_size": cacheable_chunk_size,
        "retrievable_chunk_size": retrievable_chunk_size,
        "max_subchunk_tokens": max_subchunk_tokens,
        "tokenizer_name": tokenizer_name,
        "dummy_bos_count": dummy_bos_count,
        "sentence_cache_token_format": sentence_cache_token_format,
        "deduplicate_documents_by_hash": deduplicate_documents_by_hash,
        "embedding_backend": normalize_embedding_backend(embedding_backend),
        "db_batch_size": db_batch_size,
        "embedding_device": embedding_device,
        "embedding_batch_size": embedding_batch_size,
    }


def write_db_build_manifest(db_dir: str | Path, manifest: dict[str, Any]) -> Path:
    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)
    manifest_path = db_path / DB_BUILD_MANIFEST_FILENAME
    temporary_path = db_path / f".{DB_BUILD_MANIFEST_FILENAME}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


def db_build_manifest_sha256(db_dir: str | Path) -> str:
    manifest_path = Path(db_dir) / DB_BUILD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing DB build manifest: {manifest_path}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def read_db_build_manifest(db_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(db_dir) / DB_BUILD_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != DB_BUILD_MANIFEST_FORMAT:
        raise ValueError(
            f"unsupported DB build manifest format in {manifest_path}: "
            f"{manifest.get('format')!r}"
        )
    if manifest.get("format_version") != DB_BUILD_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported DB build manifest version in {manifest_path}: "
            f"{manifest.get('format_version')!r}"
        )
    return manifest
