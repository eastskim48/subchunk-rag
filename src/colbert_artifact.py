"""Read and validate materialized ColBERT window artifacts.

General DB manifest persistence belongs to ``materialize.db_manifest``. This
module owns the ColBERT-specific reference from an artifact to that manifest,
as well as runtime access to the artifact's embedding and region sidecars.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from subchunk_store import load_subchunks, unpack_regions, validate_row
from materialize.db_manifest import (
    DB_BUILD_MANIFEST_FILENAME,
    db_build_manifest_sha256,
    read_db_build_manifest,
)

ARTIFACT_FORMAT = "colbert_window_artifact_v2"
DATA_ARTIFACT_FORMAT = "colbert_window_data_v3"
LEGACY_RESIDUAL_DATA_ARTIFACT_FORMAT = "colbert_window_data_residual_2bit_v1"
LEGACY_RESIDUAL_STORAGE_FORMAT = "colbert_residual_2bit_v1"
RESIDUAL_DATA_ARTIFACT_FORMAT = "colbert_window_data_residual_2bit_interleaved_v1"
RESIDUAL_STORAGE_FORMAT = "colbert_residual_2bit_interleaved_v1"


def colbert_2bit_token_record_dtype(embedding_dim: int) -> np.dtype:
    """Return the fixed-width interleaved record dtype for one document token."""

    residual_bytes = embedding_dim // 4
    return np.dtype(
        {
            "names": ["code", "residuals", "norm"],
            "formats": ["<i4", ("u1", (residual_bytes,)), "<f4"],
            "offsets": [0, 4, 4 + residual_bytes],
            "itemsize": 8 + residual_bytes,
        }
    )


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
    """Load the DB manifest referenced by a ColBERT artifact."""
    relative_path = reference.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("ColBERT artifact DB manifest reference requires a path")
    manifest_path = Path(artifact_dir) / relative_path
    manifest = read_db_build_manifest(manifest_path.parent)
    if manifest is None:
        raise FileNotFoundError(
            f"missing referenced DB build manifest: {manifest_path}"
        )
    return manifest


class ColBERTWindowData:
    """Memory-mapped embeddings addressed directly by integer subchunk IDs."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._decompression_profile: dict[str, float | int] | None = None
        self.index_path = self.data_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"missing ColBERT data index: {self.index_path}")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        data_format = self.index.get("format")
        self.data_format = data_format
        if data_format not in {
            DATA_ARTIFACT_FORMAT,
            LEGACY_RESIDUAL_DATA_ARTIFACT_FORMAT,
            RESIDUAL_DATA_ARTIFACT_FORMAT,
        }:
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
        if self.index.get("subchunk_id_layout") != "vector_row":
            raise ValueError(
                "ColBERT data requires integer subchunk IDs equal to vector rows"
            )
        self.num_cacheables = int(self.index["num_cacheables"])
        self.offsets = np.load(
            self.data_dir / self.index["offsets_file"], mmap_mode="r"
        )
        if (
            self.offsets.ndim != 1
            or len(self.offsets) != self.num_cacheables + 1
            or self.offsets[0] != 0
            or self.offsets[-1] != self.num_tokens
        ):
            raise ValueError("offsets do not match the subchunk/vector counts")
        self.embedding_storage = self.index.get("embedding_storage", "fp16")
        if data_format == DATA_ARTIFACT_FORMAT:
            if self.embedding_storage != "fp16":
                raise ValueError("FP16 ColBERT data requires embedding_storage='fp16'")
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
        else:
            self._initialize_residual_storage()

    def _initialize_residual_storage(self) -> None:
        expected_storage = (
            RESIDUAL_STORAGE_FORMAT
            if self.data_format == RESIDUAL_DATA_ARTIFACT_FORMAT
            else LEGACY_RESIDUAL_STORAGE_FORMAT
        )
        if self.embedding_storage != expected_storage:
            raise ValueError(
                "2-bit ColBERT data format and embedding storage mismatch: "
                f"format={self.data_format!r}, storage={self.embedding_storage!r}, "
                f"expected_storage={expected_storage!r}"
            )
        if int(self.index.get("nbits", 0)) != 2:
            raise ValueError("residual ColBERT data requires nbits=2")
        if self.embedding_dim <= 0 or self.embedding_dim % 4 != 0:
            raise ValueError(
                "2-bit ColBERT data requires embedding_dim divisible by four"
            )
        num_centroids = int(self.index.get("num_centroids", 0))
        if num_centroids <= 1:
            raise ValueError("residual ColBERT data requires num_centroids > 1")
        centroids_path = self.data_dir / self.index["centroids_file"]
        weights_path = self.data_dir / self.index["bucket_weights_file"]
        expected_sizes = {
            centroids_path: (
                num_centroids * self.embedding_dim * np.dtype(np.float16).itemsize
            ),
        }
        self.token_records = None
        if self.embedding_storage == RESIDUAL_STORAGE_FORMAT:
            record_dtype = colbert_2bit_token_record_dtype(self.embedding_dim)
            records_file = self.index.get("token_records_file")
            if not isinstance(records_file, str) or not records_file:
                raise ValueError("interleaved 2-bit data requires token_records_file")
            expected_layout = {
                "byte_order": "little",
                "record_size": record_dtype.itemsize,
                "fields": {
                    "code": {"offset": 0, "dtype": "int32"},
                    "residuals": {
                        "offset": 4,
                        "dtype": "uint8",
                        "length": self.embedding_dim // 4,
                    },
                    "norm": {
                        "offset": 4 + self.embedding_dim // 4,
                        "dtype": "float32",
                    },
                },
            }
            if self.index.get("token_record_layout") != expected_layout:
                raise ValueError("interleaved 2-bit token record layout mismatch")
            records_path = self.data_dir / records_file
            expected_sizes[records_path] = self.num_tokens * record_dtype.itemsize
        else:
            codes_path = self.data_dir / self.index["codes_file"]
            residuals_path = self.data_dir / self.index["residuals_file"]
            expected_sizes[codes_path] = self.num_tokens * np.dtype(np.int32).itemsize
            expected_sizes[residuals_path] = (
                self.num_tokens * self.embedding_dim * 2 // 8
            )
        for path, expected_size in expected_sizes.items():
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    "ColBERT residual file size mismatch: "
                    f"path={path}, actual={actual_size}, expected={expected_size}"
                )
        if self.embedding_storage == RESIDUAL_STORAGE_FORMAT:
            self.token_records = np.memmap(
                records_path,
                dtype=record_dtype,
                mode="r",
                shape=(self.num_tokens,),
            )
            self.codes = self.token_records["code"]
            self.residuals = self.token_records["residuals"]
            self.token_norms = self.token_records["norm"]
        else:
            self.codes = np.memmap(
                codes_path,
                dtype=np.int32,
                mode="r",
                shape=(self.num_tokens,),
            )
            self.residuals = np.memmap(
                residuals_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.num_tokens, self.embedding_dim // 4),
            )
            self.token_norms = None
        centroids = np.memmap(
            centroids_path,
            dtype=np.float16,
            mode="r",
            shape=(num_centroids, self.embedding_dim),
        )
        # ColBERTv2 reconstruction repeatedly addresses the shared centroid
        # table, so materialize it once instead of converting the full table
        # for every encoding-unit lookup.
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.bucket_weights = np.load(weights_path, mmap_mode="r")
        if self.bucket_weights.shape != (4,):
            raise ValueError(
                "2-bit ColBERT bucket weights must contain exactly four values"
            )
        self.empty = torch.empty((0, self.embedding_dim), dtype=torch.float32)

    def _vectors_for_row(self, start: int, end: int) -> torch.Tensor:
        if self.embedding_storage == "fp16":
            return torch.from_numpy(self.vectors[start:end]).to(torch.float16)
        from materialize.colbert_residual import decompress_2bit_rows

        if self._decompression_profile is None:
            return decompress_2bit_rows(
                codes=np.asarray(self.codes[start:end]),
                packed_residuals=np.asarray(self.residuals[start:end]),
                centroids=np.asarray(self.centroids),
                bucket_weights=np.asarray(self.bucket_weights),
            )

        row_start = time.perf_counter()
        codes = np.asarray(self.codes[start:end])
        packed_residuals = np.asarray(self.residuals[start:end])
        slice_setup_sec = time.perf_counter() - row_start
        result = decompress_2bit_rows(
            codes=codes,
            packed_residuals=packed_residuals,
            centroids=np.asarray(self.centroids),
            bucket_weights=np.asarray(self.bucket_weights),
            profile=self._decompression_profile,
        )
        profile = self._decompression_profile
        profile["row_slice_setup_sec"] = (
            profile.get("row_slice_setup_sec", 0.0) + slice_setup_sec
        )
        profile["row_total_sec"] = (
            profile.get("row_total_sec", 0.0) + time.perf_counter() - row_start
        )
        return result

    def enable_decompression_profile(self) -> None:
        if self.embedding_storage not in {
            LEGACY_RESIDUAL_STORAGE_FORMAT,
            RESIDUAL_STORAGE_FORMAT,
        }:
            raise ValueError("decompression profiling requires 2-bit residual storage")
        self._decompression_profile = {}

    def reset_decompression_profile(self) -> None:
        if self._decompression_profile is not None:
            self._decompression_profile.clear()

    def decompression_profile(self) -> dict[str, float | int]:
        return dict(self._decompression_profile or {})

    def row_for_id(self, subchunk_id):
        if subchunk_id is None:
            return None
        return validate_row(subchunk_id, self.num_cacheables)

    def vectors_for_cacheable_ids(self, cacheable_ids) -> list[torch.Tensor]:
        if self.embedding_storage == "fp16":
            vectors = []
            for cacheable_id in cacheable_ids:
                row = (
                    self.row_for_id(cacheable_id) if cacheable_id is not None else None
                )
                if row is None:
                    vectors.append(self.empty)
                    continue
                start = int(self.offsets[row])
                end = int(self.offsets[row + 1])
                if end <= start:
                    vectors.append(self.empty)
                    continue
                vectors.append(self._vectors_for_row(start, end))
            return vectors

        vectors = [self.empty] * len(cacheable_ids)
        resolved_rows = []
        for output_idx, cacheable_id in enumerate(cacheable_ids):
            row = self.row_for_id(cacheable_id) if cacheable_id is not None else None
            if row is None:
                continue
            start = int(self.offsets[row])
            end = int(self.offsets[row + 1])
            if end > start:
                resolved_rows.append((output_idx, start, end))

        group_start_idx = 0
        while group_start_idx < len(resolved_rows):
            group_end_idx = group_start_idx + 1
            group_start = resolved_rows[group_start_idx][1]
            group_end = resolved_rows[group_start_idx][2]
            while (
                group_end_idx < len(resolved_rows)
                and resolved_rows[group_end_idx][1] == group_end
            ):
                group_end = resolved_rows[group_end_idx][2]
                group_end_idx += 1

            group_rows = resolved_rows[group_start_idx:group_end_idx]
            combined = self._vectors_for_row(group_start, group_end)
            relative_start = 0
            for output_idx, start, end in group_rows:
                row_length = end - start
                vectors[output_idx] = combined[
                    relative_start : relative_start + row_length
                ]
                relative_start += row_length

            if self._decompression_profile is not None:
                profile = self._decompression_profile
                profile["decoded_encoding_unit_count"] = profile.get(
                    "decoded_encoding_unit_count", 0
                ) + len(group_rows)
                profile["coalesced_group_count"] = (
                    profile.get("coalesced_group_count", 0) + 1
                )
                if len(group_rows) > 1:
                    profile["multi_unit_group_count"] = (
                        profile.get("multi_unit_group_count", 0) + 1
                    )

            group_start_idx = group_end_idx
        return vectors

    def region_specs_for_doc(self, doc, token_budget: int):
        if self.region_token_budget != int(token_budget):
            raise ValueError(
                "runtime region budget does not match ColBERT artifact: "
                f"runtime={token_budget}, artifact={self.region_token_budget}"
            )
        cacheable_ids = [cacheable.id for cacheable in doc.cacheables if cacheable.text]
        return unpack_regions(doc.metadata, cacheable_ids)


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
        db_dir = (
            (self.artifact_dir / self.db_manifest_reference["path"]).resolve().parent
        )
        if len(load_subchunks(str(db_dir))) != self.data.num_cacheables:
            raise ValueError("subchunk list does not match ColBERT vector row count")
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

    def enable_decompression_profile(self) -> None:
        self.data.enable_decompression_profile()

    def reset_decompression_profile(self) -> None:
        self.data.reset_decompression_profile()

    def decompression_profile(self) -> dict[str, float | int]:
        return self.data.decompression_profile()

    def region_specs_for_doc(self, doc, token_budget: int):
        return self.data.region_specs_for_doc(doc, token_budget)
