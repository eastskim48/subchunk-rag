"""Direct region retrieval with the native CASS absolute-budget selector."""

import json
from pathlib import Path
import sqlite3
import time

import chromadb

from chunk import RetrievableChunk
from compressor.methods.colbert.region import ColBERTSlidingRegionCompressor
from encoder.dense import BGE_SMALL_MODEL, DenseTextEmbedder, default_query_prefix
from subchunk_store import cacheables_from_metadata, load_subchunks
from vectordb import VectorDB


class DirectRegionSelector(ColBERTSlidingRegionCompressor):
    """Reuse selection and assembly without loading a late-interaction encoder."""

    def __init__(self, source_db):
        self._initialize_token_budget(db_dir=str(source_db))
        if self.final_token_budget is None:
            raise ValueError("Direct region evaluation requires an absolute budget")
        self.global_maxsim_gate = False

    def assemble(self, hits, source_docs):
        ordered, positions, scored = [], {}, []
        candidate_ids = set()
        for region_id, distance, metadata in hits:
            cid = metadata["chunk_id"]
            if cid not in positions:
                positions[cid] = len(ordered)
                ordered.append(source_docs[cid])
            doc = source_docs[cid]
            ids = set(range(metadata["subchunk_start"], metadata["subchunk_end"]))
            indices = [i for i, unit in enumerate(doc.cacheables) if unit.id in ids]
            if {doc.cacheables[i].id for i in indices} != ids:
                raise ValueError(f"Region {region_id} references missing sentences")
            candidate_ids.update(ids)
            scored.append(
                (
                    1.0 - distance,
                    dict(
                        chunk_idx=positions[cid],
                        region_id=region_id,
                        parent_doc_id=metadata["parent_doc_id"],
                        source_cacheables=doc.cacheables,
                        selected_indices=indices,
                    ),
                )
            )
        selected, _ = self._select_sliding_regions_with_scores(
            scored, final_token_budget=self.final_token_budget
        )
        groups = {}
        for ci, block in selected:
            groups.setdefault(ci, []).append(block)
        docs = [
            self._build_region_document(doc, groups[ci])
            for ci, doc in enumerate(ordered)
            if ci in groups
        ]
        ids = [
            sid
            for doc in docs
            for block in doc.cacheables
            for sid in block.sentence_ids
        ]
        assert len(ids) == len(set(ids))
        used = sum(self._token_len_cache[sid] for sid in ids)
        assert used <= self.final_token_budget
        # With no gate/filter, omitted unique sentences mean the native selector
        # reached the budget or rejected an overflowing candidate and stopped.
        stopped = used == self.final_token_budget or set(ids) != candidate_ids
        return docs, ids, used, stopped


class DirectRegionVectorDB(VectorDB):
    def __init__(self, db_dir, source_db, trace_path, query_batch_size=24):
        path = Path(db_dir)
        manifest = json.loads((path.parent / "index_complete.json").read_text())
        if manifest["model"] != BGE_SMALL_MODEL:
            raise ValueError("Direct index must contain BGE-small vectors")
        if Path(manifest["source_db"]).resolve() != Path(source_db).resolve():
            raise ValueError("Direct index/source DB mismatch")
        self.collection = chromadb.PersistentClient(path=str(path)).get_collection(
            "doc_collection", embedding_function=None
        )
        self.count = self.collection.count()
        if self.count != manifest["index_entries"] or not self.count:
            raise ValueError("Incomplete direct index")
        self.region_store = None
        if manifest.get("storage_layout") == "vectors_and_ids_only":
            store = Path(manifest["region_store"]).resolve()
            self.region_store = sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)
        self.source = chromadb.PersistentClient(path=str(source_db)).get_collection(
            "doc_collection", embedding_function=None
        )
        self.units = load_subchunks(str(source_db))
        self.selector = DirectRegionSelector(source_db)
        self.encoder = DenseTextEmbedder(
            BGE_SMALL_MODEL, device="cpu", batch_size=query_batch_size
        )
        self.prefix = default_query_prefix(BGE_SMALL_MODEL)
        if self.prefix != manifest["query_prefix"]:
            raise ValueError("Query prefix differs from index manifest")
        self.trace_path = Path(trace_path)
        self.last_find_timings = {}

    def _source_docs(self, hits):
        ids = list(dict.fromkeys(meta["chunk_id"] for _, _, meta in hits))
        rows = self.source.get(ids=ids, include=["documents", "metadatas"])
        docs = {
            cid: RetrievableChunk(
                id=cid,
                text=text,
                metadata=meta,
                cacheables=cacheables_from_metadata(meta, self.units),
            )
            for cid, text, meta in zip(
                rows["ids"], rows["documents"], rows["metadatas"], strict=True
            )
        }
        if set(docs) != set(ids):
            raise ValueError("Direct hits reference missing source chunks")
        return docs

    def _resolve_hits(self, result):
        ids, distances = result["ids"][0], result["distances"][0]
        if self.region_store is None:
            return list(zip(ids, distances, result["metadatas"][0], strict=True))
        placeholders = ",".join("?" for _ in ids)
        rows = self.region_store.execute(
            f"select region_id,metadata from regions where region_id in ({placeholders})",
            ids,
        ).fetchall()
        metadata = {rid: json.loads(value) for rid, value in rows}
        if set(metadata) != set(ids):
            raise ValueError(
                "Direct search returned an ID absent from the region store"
            )
        return [
            (rid, distance, metadata[rid])
            for rid, distance in zip(ids, distances, strict=True)
        ]

    def find_top_k_docs(self, top_k, queries):
        # top_k is the initial search depth, not a final retrieval cap.
        if top_k <= 0:
            raise ValueError("Initial search depth must be positive")
        started = time.perf_counter()
        vectors = self.encoder.embed_texts([self.prefix + q for q in queries])
        query_seconds = time.perf_counter() - started
        search_seconds, assembly_seconds = 0.0, 0.0
        batches = []
        with self.trace_path.open("a") as trace:
            for query, vector in zip(queries, vectors, strict=True):
                depth = min(top_k, self.count)
                while True:
                    begun = time.perf_counter()
                    result = self.collection.query(
                        query_embeddings=[vector.tolist()],
                        n_results=depth,
                        include=(
                            ["distances"]
                            if self.region_store is not None
                            else ["distances", "metadatas"]
                        ),
                    )
                    search_seconds += time.perf_counter() - begun
                    hits = self._resolve_hits(result)
                    if len(hits) != depth:
                        raise ValueError("Search returned fewer hits than requested")
                    begun = time.perf_counter()
                    docs, ids, used, stopped = self.selector.assemble(
                        hits, self._source_docs(hits)
                    )
                    assembly_seconds += time.perf_counter() - begun
                    if stopped or depth == self.count:
                        break
                    depth = min(depth * 2, self.count)
                batches.append(docs)
                trace.write(
                    json.dumps(
                        dict(
                            question=query,
                            budget=self.selector.final_token_budget,
                            selected_budget_tokens=used,
                            selected_sentence_ids=ids,
                            retrieval_depth=depth,
                            retrieved_regions=[
                                dict(id=rid, distance=distance, metadata=meta)
                                for rid, distance, meta in hits
                            ],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        self.last_find_timings = dict(
            query_time=query_seconds + search_seconds,
            postprocess_time=assembly_seconds,
            cacheable_deserialize_time=0.0,
        )
        return batches

    def store(self, chunks):
        raise NotImplementedError("Direct region evaluation is read-only")
