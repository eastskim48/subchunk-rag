import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from materialize.db_manifest import (
    build_db_manifest,
    read_db_build_manifest,
    write_db_build_manifest,
)


def test_build_db_manifest_records_preprocessing_semantics():
    manifest = build_db_manifest(
        splitter="sentence",
        merger=None,
        cacheable_chunk_size=None,
        retrievable_chunk_size=512,
        max_subchunk_tokens=180,
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        dummy_bos_count=4,
        sentence_cache_token_format="legacy",
        deduplicate_documents_by_hash=True,
        embedding_backend="chroma_default",
        db_batch_size=256,
        embedding_device="cuda",
        embedding_batch_size=256,
    )

    assert manifest == {
        "format": "subchunk_db_build_manifest",
        "format_version": 1,
        "splitter": "sentence",
        "merger": None,
        "cacheable_chunk_size": None,
        "retrievable_chunk_size": 512,
        "max_subchunk_tokens": 180,
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "dummy_bos_count": 4,
        "sentence_cache_token_format": "legacy",
        "deduplicate_documents_by_hash": True,
        "embedding_backend": "default",
        "db_batch_size": 256,
        "embedding_device": "cuda",
        "embedding_batch_size": 256,
    }


def test_write_db_build_manifest_round_trips_json(tmp_path):
    manifest = {"format": "test", "retrievable_chunk_size": 512}

    manifest_path = write_db_build_manifest(tmp_path / "db", manifest)

    assert manifest_path == tmp_path / "db" / "build_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert not (tmp_path / "db" / ".build_manifest.json.tmp").exists()


def test_read_db_build_manifest_returns_none_when_missing(tmp_path):
    assert read_db_build_manifest(tmp_path / "missing-db") is None


def test_read_db_build_manifest_validates_format(tmp_path):
    db_dir = tmp_path / "db"
    write_db_build_manifest(
        db_dir,
        {
            "format": "subchunk_db_build_manifest",
            "format_version": 1,
            "retrievable_chunk_size": 512,
        },
    )

    assert read_db_build_manifest(db_dir)["retrievable_chunk_size"] == 512
