"""Original sliding-region selection backed by expanded chunk token vectors."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from compressor.methods.colbert.gaussian import ColBERTGaussianCompressor
from compressor.methods.colbert.region import (
    ColBERTSlidingRegionCompressor,
    _configured_region_group_order,
    _validate_retrieval_chunk_larger_than_region,
)
from materialize.colbert_residual import decompress_2bit_rows
from materialize.db_manifest import read_db_build_manifest
from subchunk_store import unpack_regions


class BoundaryCompleteRegionArtifact:
    """Expose finalized chunk-conditioned sentence vectors to native scoring."""

    def __init__(self, path, db_dir, region_budget):
        from materialize.expanded_chunk_materializer import ExpandedChunkArtifact

        self.reader = ExpandedChunkArtifact(path)
        expected = self.reader.index["preparation"]["source_hashes"]
        for name, digest in expected.items():
            if hashlib.sha256((Path(db_dir) / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Expanded chunk artifact DB mismatch: {name}")
        self.region_budget = region_budget
        self.retrievable_vectors_cache = {}

    def region_specs_for_doc(self, doc, token_budget):
        if token_budget != self.region_budget:
            raise ValueError("Expanded region budget mismatch")
        return unpack_regions(doc.metadata, [u.id for u in doc.cacheables if u.text])

    def prefetch_docs(self, docs):
        for doc in docs:
            self.vectors_for_doc(doc)

    def vectors_for_doc(self, doc):
        key = (str(doc.id), tuple(u.id for u in doc.cacheables))
        if key not in self.retrievable_vectors_cache:
            self.retrievable_vectors_cache[key] = self.reader.vectors_for_doc(doc)
        return self.retrievable_vectors_cache[key]


class ExpandedRegionArtifact:
    def __init__(self, owner):
        self.owner = owner
        self.retrievable_vectors_cache = {}

    def region_specs_for_doc(self, doc, token_budget):
        if token_budget != 180:
            raise ValueError("Expanded fixed-window experiment requires budget180")
        return unpack_regions(doc.metadata, [u.id for u in doc.cacheables if u.text])

    def vectors_for_doc(self, doc):
        key = (str(doc.id), tuple(u.id for u in doc.cacheables))
        if key in self.retrievable_vectors_cache:
            return self.retrievable_vectors_cache[key]
        owner = self.owner
        text, raw, sha = owner.source.execute(
            "select c.text,g.units,g.original_sha256 from chunks c join geometry g on c.id=g.id where c.id=?",
            (str(doc.id),),
        ).fetchone()
        assert hashlib.sha256(doc.text.encode()).hexdigest() == sha, doc.id
        spans = json.loads(raw)
        assert len(spans) == len(doc.cacheables), doc.id
        for span, unit in zip(spans, doc.cacheables, strict=True):
            assert span["id"] == unit.id
            assert text[span["start"] : span["end"]] == unit.text
        start, count, offsets, positions = owner.mapping.execute(
            "select start,count,offsets,positions from chunks where id=?",
            (str(doc.id),),
        ).fetchone()
        offsets = np.frombuffer(offsets, dtype=np.int32).reshape(-1, 2)
        positions = np.frombuffer(positions, dtype=np.int32)
        if count:
            rows = owner.data[start : start + count]
            vectors = decompress_2bit_rows(
                rows["code"], rows["residuals"], owner.centroids, owner.weights
            ).cpu()
        else:
            vectors = torch.empty((0, owner.centroids.shape[1]), dtype=torch.float32)
        result = []
        for span in spans:
            matching = (offsets[positions, 0] < span["end"]) & (
                offsets[positions, 1] > span["start"]
            )
            result.append(vectors[torch.from_numpy(matching)])
        self.retrievable_vectors_cache[key] = result
        return result


class ColBERTExpandedRegionCompressor(ColBERTSlidingRegionCompressor):
    def __init__(self):
        path = Path(os.environ["COLBERT_CHUNK_DIR"])
        index_path = path / "index.json"
        if (
            index_path.exists()
            and json.loads(index_path.read_text()).get("format")
            == "lateon_boundary_complete_chunks_v1"
        ):
            from compressor.methods.colbert.base import _configured_db_dir

            ColBERTSlidingRegionCompressor.__init__(self)
            self.artifact = BoundaryCompleteRegionArtifact(
                path, _configured_db_dir(), self.region_token_budget
            )
            return
        # Load exactly the expanded codec/query encoder used by the Gaussian run.
        ColBERTGaussianCompressor.__init__(self)

        stats = json.loads(
            (Path(os.environ["COLBERT_CHUNK_DIR"]) / "prepared.json").read_text()
        )
        self._initialize_token_budget(db_dir=stats["db"])
        manifest = read_db_build_manifest(stats["db"])
        import chromadb

        collection = chromadb.PersistentClient(path=stats["db"]).get_collection(
            "doc_collection", embedding_function=None
        )
        if (collection.metadata or {}).get("region_token_budget") != 180:
            raise ValueError("DB must contain existing region180 membership")
        _validate_retrieval_chunk_larger_than_region(
            manifest["retrievable_chunk_size"], 180
        )
        self.artifact = ExpandedRegionArtifact(self)
        self.region_token_budget = 180
        self.region_group_order = _configured_region_group_order()
        self._sliding_region_spec_cache = {}
        self._use_direct_2bit_maxsim = False
        self.last_profile = {}
        self.query_encoder_warmup_time = 0.0
        print(
            "Expanded fixed180: original stored regions, region MaxSim, "
            "Llama unique-unit budget, global ID dedup, add-then-check",
            flush=True,
        )
