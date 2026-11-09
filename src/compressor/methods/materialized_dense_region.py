"""Native dense region selection with previously materialized region vectors."""

import hashlib
import json
import os
from pathlib import Path

import torch

from compressor.methods.dense import DenseSlidingSubchunkCompressor


class MaterializedDenseRegionCompressor(DenseSlidingSubchunkCompressor):
    def __init__(self):
        super().__init__()
        folder = Path(os.environ["DENSE_REGION_DIR"])
        index = json.loads((folder / "index.json").read_text())
        if index["format"] != "materialized_dense_cass_regions_v1":
            raise ValueError("Unsupported materialized region format")
        if (
            index["embedding_model"] != self.embedding_model
            or index["embedding_query_prefix"] != self.query_prefix
        ):
            raise ValueError("Materialized dense model/query prefix mismatch")
        path = folder / index["embedding_file"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != index["embedding_file_sha256"]:
            raise ValueError("Materialized region embedding hash mismatch")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["format"] != index["format"]:
            raise ValueError("Materialized region payload format mismatch")
        self.stored_vectors = data["embeddings"]
        if self.stored_vectors.shape != (
            index["region_count"],
            self.embedder.embedding_dim,
        ):
            raise ValueError("Materialized region embedding shape mismatch")
        if not torch.isfinite(self.stored_vectors).all():
            raise ValueError("Nonfinite materialized region vectors")
        if len(data["region_ids"]) != len(data["region_texts"]) or len(
            data["region_ids"]
        ) != len(self.stored_vectors):
            raise ValueError("Materialized region row count mismatch")
        self.stored_rows = {
            str(rid): (row, text)
            for row, (rid, text) in enumerate(
                zip(data["region_ids"], data["region_texts"])
            )
        }
        if len(self.stored_rows) != len(self.stored_vectors):
            raise ValueError("Duplicate materialized region IDs")

    def _embed_unique_cacheables(self, cacheables):
        index, rows = {}, []
        for unit in cacheables:
            key = self._dense_cacheable_key(unit)
            if key[0] != "id":
                raise ValueError("Materialized regions require stable IDs")
            row, text = self.stored_rows[str(key[1])]
            if text != unit.text:
                raise ValueError("Materialized region text mismatch")
            if key not in index:
                index[key] = len(rows)
                rows.append(row)
        return self.stored_vectors[rows].to(torch.float32), index
