"""Explicit stored-retrieval and child-to-parent sources for native evaluation."""

import hashlib
import json
from pathlib import Path

from chunk import CacheableChunk, RetrievableChunk
from subchunk_store import cacheables_from_metadata, load_subchunks
from vectordb import ChromaDB, VectorDB


class StoredRetrievalVectorDB(VectorDB):
    def store(self, chunks):
        raise NotImplementedError("Stored retrieval is read-only")

    def __init__(self, snapshot, query_file):
        self.path = Path(snapshot)
        self.index = json.loads(Path(f"{snapshot}.index.json").read_text())
        if (
            hashlib.sha256(Path(query_file).read_bytes()).hexdigest()
            != self.index["query_file_sha256"]
        ):
            raise ValueError("Stored retrieval query file mismatch")
        if (
            hashlib.sha256(self.path.read_bytes()).hexdigest()
            != self.index["snapshot_sha256"]
        ):
            raise ValueError("Stored retrieval content hash mismatch")
        self.handle = self.path.open()
        self.warmup = True
        self.last_find_timings = {}

    def find_top_k_docs(self, top_k, queries):
        if top_k != self.index["top_k"]:
            raise ValueError("Stored retrieval TopK mismatch")
        records = [json.loads(next(self.handle)) for _ in queries]
        if self.warmup:
            if len(queries) != 1:
                raise ValueError("Expected native single-query warmup")
            self.handle.seek(0)
            self.warmup = False
        if [r["query"] for r in records] != queries:
            raise ValueError("Stored retrieval row order mismatch")
        return [
            [
                RetrievableChunk(
                    **{k: v for k, v in doc.items() if k != "cacheables"},
                    cacheables=[
                        CacheableChunk.from_payload(u) for u in doc["cacheables"]
                    ],
                )
                for doc in record["docs"]
            ]
            for record in records
        ]


class ParentChunkVectorDB(VectorDB):
    def store(self, chunks):
        raise NotImplementedError("Parent retrieval is read-only")

    def __init__(self, child_db_dir, mapping_file):
        import chromadb

        self.mapping = json.loads(Path(mapping_file).read_text())
        if Path(child_db_dir).resolve() != Path(self.mapping["child_db_dir"]).resolve():
            raise ValueError("Parent mapping child DB mismatch")
        self.child = ChromaDB(child_db_dir)
        self.parent_dir = self.mapping["parent_db_dir"]
        self.parent = chromadb.PersistentClient(path=self.parent_dir).get_collection(
            "doc_collection", embedding_function=None
        )
        self.subchunks = load_subchunks(self.parent_dir)
        self.last_find_timings = {}

    def find_top_k_docs(self, top_k, queries):
        batches = self.child.find_top_k_docs(top_k, queries)
        result = []
        for docs in batches:
            ids = []
            for doc in docs:
                parent = self.mapping["links"][str(doc.id)]
                if parent not in ids:
                    ids.append(parent)
                if len(ids) == self.mapping["max_parents"]:
                    break
            rows = self.parent.get(ids=ids, include=["documents", "metadatas"])
            by_id = {
                cid: (text, metadata)
                for cid, text, metadata in zip(
                    rows["ids"], rows["documents"], rows["metadatas"]
                )
            }
            result.append(
                [
                    RetrievableChunk(
                        id=cid,
                        text=by_id[cid][0],
                        cacheables=cacheables_from_metadata(
                            by_id[cid][1], self.subchunks
                        ),
                        metadata=by_id[cid][1],
                    )
                    for cid in ids
                ]
            )
        self.last_find_timings = self.child.last_find_timings
        return result
