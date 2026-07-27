"""Read and validate materialized ColBERT window artifacts.

General DB manifest persistence belongs to ``materialize.db_manifest``. This
module owns the ColBERT-specific reference from an artifact to that manifest,
as well as runtime access to the artifact's embedding and region sidecars.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colbert_metadata import read_json_with_sha256
from materialize.db_manifest import (
    DB_BUILD_MANIFEST_FILENAME,
    db_build_manifest_sha256,
    read_db_build_manifest,
)

ARTIFACT_FORMAT = "colbert_window_artifact_v2"
DATA_ARTIFACT_FORMAT = "colbert_window_data_v3"


def build_db_manifest_reference(
    *, db_dir: str | Path, artifact_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the path-and-hash reference stored in a ColBERT artifact index."""
    db_path = Path(db_dir)
    manifest = read_db_build_manifest(db_path)
    if manifest is None:
        raise FileNotFoundError(
            f"missing DB build manifest: {db_path / DB_BUILD_MANIFEST_FILENAME}"
        )
    manifest_path = db_path / DB_BUILD_MANIFEST_FILENAME
    reference = {
        "path": os.path.relpath(manifest_path, start=Path(artifact_dir)),
        "sha256": db_build_manifest_sha256(db_path),
    }
    return manifest, reference


def read_referenced_db_manifest(
    *, artifact_dir: str | Path, reference: dict[str, Any]
) -> dict[str, Any]:
    """Load a ColBERT artifact's DB manifest after verifying its content hash."""
    relative_path = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("ColBERT artifact DB manifest reference requires a path")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("ColBERT artifact DB manifest reference requires a SHA-256")
    manifest_path = Path(artifact_dir) / relative_path
    manifest = read_db_build_manifest(manifest_path.parent)
    if manifest is None:
        raise FileNotFoundError(
            f"missing referenced DB build manifest: {manifest_path}"
        )
    actual_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "referenced DB build manifest SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}, "
            f"path={manifest_path}"
        )
    return manifest


class FixedChunkColBERTArtifact:
    """Read token vectors materialized directly for fixed retrieval chunks."""

    FORMAT = "fixed_chunk_colbert_artifact_v1"

    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        index_path = os.path.join(artifact_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"missing fixed-chunk ColBERT artifact index: {index_path}"
            )
        with open(index_path, "r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("format") != self.FORMAT:
            raise ValueError(
                "unsupported fixed-chunk ColBERT artifact format: "
                f"{self.index.get('format')}"
            )
        if int(self.index.get("truncated_count", -1)) != 0:
            raise ValueError(
                "fixed-chunk ColBERT artifact has truncated chunks; rebuild with "
                "--doc-maxlen 512 --segment-long-docs"
            )
        self.embedding_dim = int(self.index["embedding_dim"])
        self.num_tokens = int(self.index["num_tokens"])
        self.id_to_row = self.index["id_to_row"]
        self.offsets = np.load(
            os.path.join(artifact_dir, self.index["offsets_file"]), mmap_mode="r"
        )
        self.vectors = np.memmap(
            os.path.join(artifact_dir, self.index["vectors_file"]),
            dtype=np.float16,
            mode="r",
            shape=(self.num_tokens, self.embedding_dim),
        )
        self.empty = torch.empty((0, self.embedding_dim), dtype=torch.float16)

    def vectors_for_chunk_id(self, chunk_id: str) -> torch.Tensor:
        row = self.id_to_row.get(str(chunk_id))
        if row is None:
            return self.empty
        start = int(self.offsets[row])
        end = int(self.offsets[row + 1])
        if end <= start:
            return self.empty
        return torch.from_numpy(self.vectors[start:end]).to(torch.float16)


class ColBERTWindowData:
    """Memory-mapped embeddings with split JSON metadata preloaded in memory."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"missing ColBERT data index: {self.index_path}")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != DATA_ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported ColBERT data format: " f"{self.index.get('format')}"
            )
        self.embedding_dim = int(self.index["embedding_dim"])
        self.num_tokens = int(self.index["num_tokens"])
        self.region_token_budget = self.index.get("region_token_budget")
        if not isinstance(self.region_token_budget, int) or isinstance(
            self.region_token_budget, bool
        ):
            raise ValueError("ColBERT data requires an integer region_token_budget")
        rows_file = self.index.get("cacheable_rows_file")
        rows_sha256 = self.index.get("cacheable_rows_sha256")
        regions_file = self.index.get("region_payloads_file")
        regions_sha256 = self.index.get("region_payloads_sha256")
        for field_name, value in (
            ("cacheable_rows_file", rows_file),
            ("cacheable_rows_sha256", rows_sha256),
            ("region_payloads_file", regions_file),
            ("region_payloads_sha256", regions_sha256),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"ColBERT data requires {field_name}")
        self.id_to_row = read_json_with_sha256(self.data_dir / rows_file, rows_sha256)
        self.region_payloads_by_chunk = read_json_with_sha256(
            self.data_dir / regions_file, regions_sha256
        )
        expected_cacheables = int(self.index["num_cacheables"])
        if len(self.id_to_row) != expected_cacheables:
            raise ValueError(
                "ColBERT cacheable-row count does not match data index: "
                f"metadata={len(self.id_to_row)}, "
                f"index={expected_cacheables}"
            )
        expected_region_chunks = int(self.index["region_spec_chunk_count"])
        if len(self.region_payloads_by_chunk) != expected_region_chunks:
            raise ValueError(
                "ColBERT region-payload count does not match data index: "
                f"metadata={len(self.region_payloads_by_chunk)}, "
                f"index={expected_region_chunks}"
            )
        self.offsets = np.load(
            self.data_dir / self.index["offsets_file"], mmap_mode="r"
        )
        vectors_path = self.data_dir / self.index["vectors_file"]
        # Each cacheable maps to one contiguous [offsets[row], offsets[row + 1])
        # slice in this flattened token-embedding matrix.
        self.vectors = np.memmap(
            vectors_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_tokens, self.embedding_dim),
        )
        self.empty = torch.empty((0, self.embedding_dim), dtype=torch.float16)

    def vectors_for_cacheable_ids(self, cacheable_ids) -> list[torch.Tensor]:
        vectors = []
        for cacheable_id in cacheable_ids:
            row = (
                self.id_to_row.get(str(cacheable_id))
                if cacheable_id is not None
                else None
            )
            if row is None:
                vectors.append(self.empty)
                continue
            start = int(self.offsets[row])
            end = int(self.offsets[row + 1])
            if end <= start:
                vectors.append(self.empty)
                continue
            array = self.vectors[start:end]
            vectors.append(torch.from_numpy(array).to(torch.float16))
        return vectors

    def region_specs_for_doc(self, doc, token_budget: int):
        if self.region_token_budget != int(token_budget):
            raise ValueError(
                "runtime region budget does not match ColBERT artifact: "
                f"runtime={token_budget}, artifact={self.region_token_budget}"
            )
        doc_id = str(getattr(doc, "id", ""))
        payload = self.region_payloads_by_chunk.get(doc_id)
        if payload is None:
            raise ValueError(f"ColBERT region specs missing runtime chunk: {doc_id}")
        cacheable_ids = [
            getattr(cacheable, "id", None)
            for cacheable in getattr(doc, "cacheables", []) or []
            if getattr(cacheable, "text", None)
        ]
        expected_cacheable_ids = payload.get("cacheable_ids")
        if expected_cacheable_ids != cacheable_ids:
            raise ValueError(
                "runtime cacheable IDs do not match ColBERT region specs: "
                f"chunk={doc_id}, expected={expected_cacheable_ids!r}, "
                f"runtime={cacheable_ids!r}"
            )
        return [
            (int(item[0]), tuple(int(idx) for idx in item[1]))
            for item in payload.get("specs", [])
        ]


class ColBERTWindowArtifact:
    """Validated top-level artifact index and runtime data accessor."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.index_path = self.artifact_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"missing ColBERT window artifact index: {self.index_path}"
            )
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        artifact_format = self.index.get("format")
        if artifact_format != ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported ColBERT window artifact format: "
                f"{artifact_format}; rebuild with the official ColBERT path"
            )
        self.db_manifest_reference = self.index.get("db_manifest")
        if not isinstance(self.db_manifest_reference, dict):
            raise ValueError("ColBERT artifact requires a DB manifest reference")
        self.db_manifest = read_referenced_db_manifest(
            artifact_dir=self.artifact_dir,
            reference=self.db_manifest_reference,
        )
        # Retrieval windows can repeat the same ordered cacheable list across
        # queries, so keep the already-created tensor views for reuse.
        self.retrievable_vectors_cache: dict[
            tuple[str, tuple[str | None, ...]], list[torch.Tensor]
        ] = {}
        data_dir = self.index.get("data_dir")
        if not isinstance(data_dir, str) or not data_dir:
            raise ValueError("ColBERT artifact requires a data_dir")
        self.data = ColBERTWindowData(self.artifact_dir / data_dir)
        window_token_budget = self.index.get("window_token_budget")
        if not isinstance(window_token_budget, int) or isinstance(
            window_token_budget, bool
        ):
            raise ValueError("ColBERT artifact requires an integer window_token_budget")
        if self.data.region_token_budget != window_token_budget:
            raise ValueError(
                "ColBERT region budget does not match artifact window budget: "
                f"region={self.data.region_token_budget}, window={window_token_budget}"
            )

    def validate_db_manifest(self, db_dir: str | Path) -> None:
        referenced_path = (
            self.artifact_dir / self.db_manifest_reference["path"]
        ).resolve()
        runtime_path = (Path(db_dir) / DB_BUILD_MANIFEST_FILENAME).resolve()
        if runtime_path != referenced_path:
            raise ValueError(
                "runtime DB manifest path does not match ColBERT artifact reference: "
                f"artifact={referenced_path}, runtime={runtime_path}"
            )
        expected_sha256 = self.db_manifest_reference["sha256"]
        actual_sha256 = db_build_manifest_sha256(db_dir)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "runtime DB build manifest does not match ColBERT artifact: "
                f"artifact={expected_sha256}, runtime={actual_sha256}, db_dir={db_dir}"
            )

    def vectors_for_doc(self, doc) -> list[torch.Tensor]:
        cacheable_ids = tuple(
            getattr(cacheable, "id", None)
            for cacheable in getattr(doc, "cacheables", []) or []
        )
        cache_key = (str(getattr(doc, "id", "")), cacheable_ids)
        cached = self.retrievable_vectors_cache.get(cache_key)
        if cached is not None:
            return cached
        vectors = self.data.vectors_for_cacheable_ids(cacheable_ids)
        self.retrievable_vectors_cache[cache_key] = vectors
        return vectors

    def region_specs_for_doc(self, doc, token_budget: int):
        return self.data.region_specs_for_doc(doc, token_budget)
