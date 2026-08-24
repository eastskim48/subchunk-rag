"""Dense-embedding baseline for token-budgeted subchunk selection."""

import os
import time
from typing import List

import torch

from chunk import CacheableChunk, RetrievableChunk
from colbert_artifact import ColBERTWindowArtifact
from compressor.base import Compressor
from compressor.methods.colbert.base import (
    _configured_db_dir,
    _resolve_configured_retrieval_chunk_size,
)
from compressor.methods.colbert.region import (
    ColBERTSlidingRegionCompressor,
    _configured_region_group_order,
    _validate_retrieval_chunk_larger_than_region,
)
from compressor.token_budget import TokenBudgetMixin
from encoder.dense import BGE_M3_MODEL, DenseTextEmbedder, default_query_prefix

QUERY_PREFIX = default_query_prefix(BGE_M3_MODEL)


class _DenseEmbeddingSelector(TokenBudgetMixin, Compressor):
    """Own the dense query encoder shared by dense selection variants."""

    def __init__(self):
        super().__init__()
        print("Dense selection enabled. Initializing local embedding model...")
        self.embedding_model = os.getenv("DENSE_EMBED_MODEL", BGE_M3_MODEL)
        self.device = os.getenv("DENSE_EMBED_DEVICE", "cpu")
        self.batch_size = int(os.getenv("DENSE_EMBED_BATCH_SIZE", "128"))
        self.query_prefix = default_query_prefix(self.embedding_model)
        self.embedder = DenseTextEmbedder(
            model_name=self.embedding_model,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.device = self.embedder.device
        self._runtime_cacheable_embeddings: dict[str, tuple[str, torch.Tensor]] = {}
        self.runtime_cache_hits = 0
        self.runtime_cache_misses = 0

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        return self.embedder.embed_texts(texts).to(self.device)

    def _format_query(self, query: str) -> str:
        return f"{self.query_prefix}{query}"

    @staticmethod
    def _dense_cacheable_key(cacheable: CacheableChunk) -> tuple[str, str | int]:
        cacheable_id = getattr(cacheable, "id", None)
        if cacheable_id:
            return ("id", str(cacheable_id))
        return ("anonymous", id(cacheable))

    def _embed_unique_cacheables(
        self, cacheables: List[CacheableChunk]
    ) -> tuple[torch.Tensor, dict[tuple[str, str | int], int]]:
        """Embed each stable candidate ID once and retain vectors across batches."""

        cache = getattr(self, "_runtime_cacheable_embeddings", None)
        if cache is None:
            cache = {}
            self._runtime_cacheable_embeddings = cache
            self.runtime_cache_hits = 0
            self.runtime_cache_misses = 0

        unique_keys: list[tuple[str, str | int]] = []
        text_by_key: dict[tuple[str, str | int], str] = {}
        cacheable_index_by_key: dict[tuple[str, str | int], int] = {}
        for cacheable in cacheables:
            key = self._dense_cacheable_key(cacheable)
            text = cacheable.text
            existing_text = text_by_key.get(key)
            if existing_text is not None and existing_text != text:
                raise ValueError(
                    "dense runtime cacheable ID has conflicting texts: " f"{key[1]!r}"
                )
            if key not in cacheable_index_by_key:
                cacheable_index_by_key[key] = len(unique_keys)
                unique_keys.append(key)
                text_by_key[key] = text

        embedding_by_key: dict[tuple[str, str | int], torch.Tensor] = {}
        missing_keys: list[tuple[str, str | int]] = []
        missing_texts: list[str] = []
        for key in unique_keys:
            if key[0] != "id":
                missing_keys.append(key)
                missing_texts.append(text_by_key[key])
                continue
            cached = cache.get(str(key[1]))
            if cached is None:
                missing_keys.append(key)
                missing_texts.append(text_by_key[key])
                continue
            cached_text, cached_embedding = cached
            if cached_text != text_by_key[key]:
                raise ValueError(
                    "dense runtime cacheable ID has conflicting texts: " f"{key[1]!r}"
                )
            embedding_by_key[key] = cached_embedding
            self.runtime_cache_hits += 1

        if missing_texts:
            missing_embeddings = (
                self._embed_texts(missing_texts).detach().cpu().to(torch.float32)
            )
            if missing_embeddings.shape[0] != len(missing_keys):
                raise ValueError(
                    "dense embedder returned an unexpected row count: "
                    f"expected={len(missing_keys)}, "
                    f"actual={missing_embeddings.shape[0]}"
                )
            for key, text, embedding in zip(
                missing_keys, missing_texts, missing_embeddings
            ):
                embedding_by_key[key] = embedding
                if key[0] == "id":
                    cache[str(key[1])] = (text, embedding)
                    self.runtime_cache_misses += 1

        embeddings = (
            torch.stack([embedding_by_key[key] for key in unique_keys])
            if unique_keys
            else torch.empty((0, 0), dtype=torch.float32)
        )
        return embeddings, cacheable_index_by_key


class _TokenBudgetDenseSelector(_DenseEmbeddingSelector):
    """Apply a shared token budget after ranking dense sentence candidates."""

    def __init__(self):
        self._initialize_token_budget()
        super().__init__()

    def _build_unselected_document(self, doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned

    @staticmethod
    def _deduplicate_sentence_refs(
        chunks: List[RetrievableChunk],
        sentence_refs: List[tuple[int, int]],
        similarities: torch.Tensor,
    ) -> tuple[List[tuple[int, int]], torch.Tensor]:
        dedup_refs, dedup_scores = (
            _TokenBudgetDenseSelector._deduplicate_sentence_refs_multi_scores(
                chunks=chunks,
                sentence_refs=sentence_refs,
                score_tensors=[similarities],
            )
        )
        if not dedup_scores:
            return dedup_refs, torch.empty(0, dtype=similarities.dtype)
        return dedup_refs, dedup_scores[0]

    @staticmethod
    def _deduplicate_sentence_refs_multi_scores(
        chunks: List[RetrievableChunk],
        sentence_refs: List[tuple[int, int]],
        score_tensors: List[torch.Tensor],
    ) -> tuple[List[tuple[int, int]], List[torch.Tensor]]:
        dedup_refs: List[tuple[int, int]] = []
        dedup_score_lists: List[List[torch.Tensor]] = [[] for _ in score_tensors]
        index_by_cacheable_id: dict[str, int] = {}

        for flat_idx, (chunk_idx, sentence_idx) in enumerate(sentence_refs):
            cacheables = getattr(chunks[chunk_idx], "cacheables", []) or []
            cacheable = (
                cacheables[sentence_idx]
                if 0 <= sentence_idx < len(cacheables)
                else None
            )
            cacheable_id = getattr(cacheable, "id", None)
            scores = [score_tensor[flat_idx] for score_tensor in score_tensors]
            primary_score = sum(score.to(torch.float32) for score in scores)

            if cacheable_id:
                existing_idx = index_by_cacheable_id.get(cacheable_id)
                if existing_idx is not None:
                    existing_primary = sum(
                        score_list[existing_idx].to(torch.float32)
                        for score_list in dedup_score_lists
                    )
                    if primary_score > existing_primary:
                        dedup_refs[existing_idx] = (chunk_idx, sentence_idx)
                        for score_list, score in zip(dedup_score_lists, scores):
                            score_list[existing_idx] = score
                    continue
                index_by_cacheable_id[cacheable_id] = len(dedup_refs)

            dedup_refs.append((chunk_idx, sentence_idx))
            for score_list, score in zip(dedup_score_lists, scores):
                score_list.append(score)

        dedup_tensors = [
            (
                torch.stack(score_list).to(score_tensors[idx].dtype)
                if score_list
                else torch.empty(0, dtype=score_tensors[idx].dtype)
            )
            for idx, score_list in enumerate(dedup_score_lists)
        ]
        return dedup_refs, dedup_tensors

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, batch_similarities = (
            self._score_chunk_texts_for_batch(
                batch_top_k_docs,
                batch_queries,
            )
        )
        summarized_batches = []
        cursor = 0
        for chunks, chunk_texts_per_doc in zip(
            batch_top_k_docs, batch_chunk_texts_per_doc
        ):
            summarized_chunks = [
                self._build_unselected_document(chunk) for chunk in chunks
            ]
            if batch_similarities is None:
                summarized_batches.append(summarized_chunks)
                continue

            sentence_refs = []
            for chunk_idx, chunk_texts in enumerate(chunk_texts_per_doc):
                for sentence_idx in range(len(chunk_texts)):
                    sentence_refs.append((chunk_idx, sentence_idx))

            similarities = batch_similarities[cursor : cursor + len(sentence_refs)]
            cursor += len(sentence_refs)
            sentence_refs, similarities = self._deduplicate_sentence_refs(
                chunks=chunks,
                sentence_refs=sentence_refs,
                similarities=similarities,
            )

            final_token_budget = self._resolve_final_token_budget(chunks)
            ranked_indices = torch.argsort(similarities, descending=True).tolist()
            selected_by_doc = {}
            used_tokens = 0
            for flat_idx in ranked_indices:
                chunk_idx, sentence_idx = sentence_refs[flat_idx]
                selected_by_doc.setdefault(chunk_idx, []).append(sentence_idx)
                cacheable = chunks[chunk_idx].cacheables[sentence_idx]
                used_tokens += self._cacheable_token_len(cacheable)
                if used_tokens >= final_token_budget:
                    break

            for chunk_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[chunk_idx] = self._build_selected_document(
                    chunks[chunk_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)
        return summarized_batches


class DenseCompressor(_TokenBudgetDenseSelector):
    """Score retrieved subchunks against materialized dense document vectors."""

    def __init__(self):
        super().__init__()
        self.dense_embed_dir = os.getenv("DENSE_EMBED_DIR")
        if not self.dense_embed_dir:
            raise ValueError("DENSE_EMBED_DIR must be set for compress_method='dense'")
        self.dense_embed_payload = self._load_dense_embed_payload()
        self.raw_embedding_pool = {}

    def _dense_embed_path(self) -> str:
        return os.path.join(self.dense_embed_dir, "dense_embed.pt")

    def _load_dense_embed_payload(self):
        payload_path = self._dense_embed_path()
        if not os.path.exists(payload_path):
            raise FileNotFoundError(
                f"materialized dense embedding file missing: {payload_path}"
            )
        payload = torch.load(payload_path, map_location="cpu")
        if payload.get("format") != "dense_embed_single_v1":
            raise ValueError(f"unsupported dense embed format: {payload.get('format')}")
        payload_model = payload.get("embedding_model")
        if payload_model != self.embedding_model:
            raise ValueError(
                f"dense embed model mismatch: payload={payload_model!r} runtime={self.embedding_model!r}"
            )
        payload_query_prefix = payload.get("embedding_query_prefix", QUERY_PREFIX)
        if payload_query_prefix != self.query_prefix:
            raise ValueError(
                "dense embed query prefix mismatch: "
                f"payload={payload_query_prefix!r} runtime={self.query_prefix!r}"
            )

        return payload

    def _get_doc_index_entry(self, doc_id: str):
        return self.dense_embed_payload.get("doc_index", {}).get(doc_id)

    @staticmethod
    def _select_rows_by_chunk_ids(
        embeddings: torch.Tensor,
        materialized_chunk_ids: List[str],
        expected_chunk_ids: List[str],
        doc_id: str,
    ) -> torch.Tensor:
        index_by_chunk_id = {
            chunk_id: idx for idx, chunk_id in enumerate(materialized_chunk_ids)
        }
        missing_chunk_ids = [
            chunk_id
            for chunk_id in expected_chunk_ids
            if chunk_id not in index_by_chunk_id
        ]
        if missing_chunk_ids:
            raise KeyError(
                f"materialized dense embedding chunk_ids missing for {doc_id}: {missing_chunk_ids[:5]}"
            )
        row_indices = [index_by_chunk_id[chunk_id] for chunk_id in expected_chunk_ids]
        return embeddings[row_indices]

    def _load_materialized_doc_embeddings(self, doc: RetrievableChunk) -> torch.Tensor:
        if doc.id in self.raw_embedding_pool:
            return self.raw_embedding_pool[doc.id]

        expected_chunk_ids = [
            cacheable.id for cacheable in getattr(doc, "cacheables", [])
        ]
        doc_index_entry = self._get_doc_index_entry(doc.id)
        source_doc_id = doc.id
        if doc_index_entry is None:
            source_doc_id = doc.metadata.get("parent_doc_id", doc.id)
            doc_index_entry = self._get_doc_index_entry(source_doc_id)
        if doc_index_entry is None:
            raise KeyError(f"materialized dense embedding index missing doc {doc.id}")

        row_start = int(doc_index_entry["row_start"])
        row_end = int(doc_index_entry["row_end"])
        embeddings = self.dense_embed_payload["embeddings"][row_start:row_end]
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"materialized dense embeddings for {doc.id} are not a tensor"
            )
        embeddings = embeddings.to(torch.float32)
        materialized_chunk_ids = doc_index_entry.get("chunk_ids")
        if isinstance(expected_chunk_ids, list) and isinstance(
            materialized_chunk_ids, list
        ):
            if materialized_chunk_ids != expected_chunk_ids:
                embeddings = self._select_rows_by_chunk_ids(
                    embeddings=embeddings,
                    materialized_chunk_ids=materialized_chunk_ids,
                    expected_chunk_ids=expected_chunk_ids,
                    doc_id=doc.id,
                )
        self.raw_embedding_pool[doc.id] = embeddings
        return embeddings

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc = []
        flattened_embeddings = []
        sentence_owner_query_indices = []

        for query_idx, docs in enumerate(batch_top_k_docs):
            chunk_texts_per_doc = []
            for doc in docs:
                chunk_texts = [
                    cacheable.text
                    for cacheable in getattr(doc, "cacheables", [])
                    if cacheable.text
                ]
                if chunk_texts:
                    chunk_texts_per_doc.append(chunk_texts)
                    flattened_embeddings.append(
                        self._load_materialized_doc_embeddings(doc)
                    )
                    sentence_owner_query_indices.extend([query_idx] * len(chunk_texts))
                else:
                    chunk_texts_per_doc.append([])
            batch_chunk_texts_per_doc.append(chunk_texts_per_doc)

        if not flattened_embeddings:
            return batch_chunk_texts_per_doc, None

        query_texts = [self._format_query(query) for query in batch_queries]
        query_embeddings = (
            self._embed_texts(query_texts).detach().cpu().to(torch.float32)
        )
        sentence_embeddings = torch.cat(flattened_embeddings, dim=0)
        owner_query_tensor = torch.tensor(
            sentence_owner_query_indices, dtype=torch.long
        )
        similarities = (sentence_embeddings * query_embeddings[owner_query_tensor]).sum(
            dim=1
        )
        return batch_chunk_texts_per_doc, similarities


class DenseOnlineCompressor(_TokenBudgetDenseSelector):
    """Score retrieved subchunks by embedding candidates at evaluation time."""

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc = []
        flattened_cacheables = []
        sentence_owner_query_indices = []

        for query_idx, docs in enumerate(batch_top_k_docs):
            chunk_texts_per_doc = []
            for doc in docs:
                document_cacheables = [
                    cacheable
                    for cacheable in getattr(doc, "cacheables", [])
                    if cacheable.text
                ]
                chunk_texts = [cacheable.text for cacheable in document_cacheables]
                chunk_texts_per_doc.append(chunk_texts)
                flattened_cacheables.extend(document_cacheables)
                sentence_owner_query_indices.extend([query_idx] * len(chunk_texts))
            batch_chunk_texts_per_doc.append(chunk_texts_per_doc)

        if not flattened_cacheables:
            return batch_chunk_texts_per_doc, None

        query_texts = [self._format_query(query) for query in batch_queries]
        query_embeddings = self._embed_texts(query_texts).detach().to(torch.float32)
        unique_embeddings, cacheable_index_by_key = self._embed_unique_cacheables(
            flattened_cacheables
        )
        occurrence_rows = torch.tensor(
            [
                cacheable_index_by_key[self._dense_cacheable_key(cacheable)]
                for cacheable in flattened_cacheables
            ],
            dtype=torch.long,
            device=query_embeddings.device,
        )
        sentence_embeddings = unique_embeddings.to(query_embeddings.device)[
            occurrence_rows
        ]
        owner_query_tensor = torch.tensor(
            sentence_owner_query_indices,
            dtype=torch.long,
            device=query_embeddings.device,
        )
        similarities = (sentence_embeddings * query_embeddings[owner_query_tensor]).sum(
            dim=1
        )
        similarities = similarities.detach().cpu()
        return batch_chunk_texts_per_doc, similarities


class _DenseSlidingRegionCompressor(
    _DenseEmbeddingSelector, ColBERTSlidingRegionCompressor
):
    """Aggregate runtime dense subchunk scores over stored sliding regions."""

    region_score_aggregation: str

    def __init__(self):
        Compressor.__init__(self)
        self._initialize_token_budget()
        self.last_profile: dict[str, float | int] = {}
        self._sliding_region_spec_cache = {}
        self.region_group_order = _configured_region_group_order()

        self.embedding_model = os.getenv("DENSE_EMBED_MODEL", BGE_M3_MODEL)
        self.device = os.getenv("DENSE_EMBED_DEVICE", "cpu")
        self.batch_size = int(os.getenv("DENSE_EMBED_BATCH_SIZE", "128"))
        self.query_prefix = default_query_prefix(self.embedding_model)
        self.embedder = DenseTextEmbedder(
            model_name=self.embedding_model,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.device = self.embedder.device
        self._runtime_cacheable_embeddings: dict[str, tuple[str, torch.Tensor]] = {}
        self.runtime_cache_hits = 0
        self.runtime_cache_misses = 0

        artifact_dir = os.getenv("COLBERT_WINDOW_DIR")
        if not artifact_dir:
            dataset_path = os.getenv("DATASET_PATH")
            data_subdir = os.getenv("DATA_SUBDIR", "sent")
            if not dataset_path:
                raise ValueError(
                    "COLBERT_WINDOW_DIR or DATASET_PATH must be set for dense "
                    "sliding-region compression"
                )
            artifact_dir = os.path.join(dataset_path, data_subdir, "colbert_window")

        self.artifact = ColBERTWindowArtifact(artifact_dir)
        db_dir = _configured_db_dir()
        if db_dir is None:
            raise ValueError(
                "DB_DIR or DATASET_PATH/DATA_SUBDIR is required to validate the "
                "sliding-region artifact"
            )
        self.artifact.validate_db_manifest(db_dir)
        self.region_token_budget = int(self.artifact.index["window_token_budget"])
        retrieval_chunk_size = _resolve_configured_retrieval_chunk_size()
        _validate_retrieval_chunk_larger_than_region(
            retrieval_chunk_size, self.region_token_budget
        )
        print(
            "Dense sliding-region compression enabled. "
            f"aggregation={self.region_score_aggregation}, "
            f"artifact={artifact_dir!r}"
        )

    def warmup_query_encoder(self) -> float:
        start = time.perf_counter()
        self._embed_texts([self._format_query("warmup query")])
        self.query_encoder_warmup_time = time.perf_counter() - start
        return self.query_encoder_warmup_time

    def clear_inter_batch_cache(self) -> None:
        self._sliding_region_spec_cache.clear()

    def _sliding_regions_for_doc(
        self,
        doc: RetrievableChunk,
        chunk_idx: int,
        profile: dict[str, float | int] | None = None,
    ):
        cacheables = [
            cacheable
            for cacheable in getattr(doc, "cacheables", []) or []
            if cacheable.text
        ]
        if not cacheables:
            return []
        parent_doc_id = self._source_parent_document_id(doc, cacheables)
        region_specs = self._cached_sliding_region_specs(doc, cacheables)
        regions = [
            {
                "chunk_idx": chunk_idx,
                "center_idx": center_idx,
                "cacheable": None,
                "region_id": f"{doc.id}::dense_sliding_region_{center_idx}",
                "parent_doc_id": parent_doc_id,
                "source_doc": doc,
                "selected_indices": selected_indices,
                "source_cacheables": cacheables,
            }
            for center_idx, selected_indices in region_specs
            if selected_indices
        ]
        self._profile_add(profile, "region_count", len(regions))
        return regions

    def _score_dense_regions_for_batch(self, batch_regions, batch_queries):
        cacheable_index_by_object = {}
        referenced_cacheables = []

        for regions in batch_regions:
            for region in regions:
                for source_idx in region["selected_indices"]:
                    source_cacheables = region["source_cacheables"]
                    if not (0 <= source_idx < len(source_cacheables)):
                        continue
                    cacheable = source_cacheables[source_idx]
                    referenced_cacheables.append(cacheable)

        if not referenced_cacheables:
            return [[] for _ in batch_regions]

        query_embeddings = (
            self._embed_texts([self._format_query(query) for query in batch_queries])
            .detach()
            .to(torch.float32)
        )
        cacheable_embeddings, cacheable_index_by_key = self._embed_unique_cacheables(
            referenced_cacheables
        )
        for cacheable in referenced_cacheables:
            cacheable_index_by_object[id(cacheable)] = cacheable_index_by_key[
                self._dense_cacheable_key(cacheable)
            ]
        similarities = torch.matmul(
            query_embeddings,
            cacheable_embeddings.to(query_embeddings.device).T,
        )
        similarities = similarities.detach().cpu()

        batch_scores = []
        for query_idx, regions in enumerate(batch_regions):
            region_scores = []
            for region in regions:
                indices = [
                    cacheable_index_by_object[id(region["source_cacheables"][idx])]
                    for idx in region["selected_indices"]
                    if 0 <= idx < len(region["source_cacheables"])
                ]
                if not indices:
                    region_scores.append(float("-inf"))
                    continue
                member_scores = similarities[query_idx, indices]
                if self.region_score_aggregation == "max":
                    region_score = member_scores.max()
                elif self.region_score_aggregation == "avg":
                    region_score = member_scores.mean()
                else:
                    raise ValueError(
                        "unsupported dense region score aggregation: "
                        f"{self.region_score_aggregation!r}"
                    )
                region_scores.append(float(region_score.item()))
            batch_scores.append(region_scores)
        return batch_scores

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_regions = []
        for docs in batch_top_k_docs:
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx))
            batch_regions.append(regions)

        batch_region_scores = self._score_dense_regions_for_batch(
            batch_regions, batch_queries
        )
        summarized_batches = []
        for docs, regions, region_scores in zip(
            batch_top_k_docs, batch_regions, batch_region_scores
        ):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(docs)
            scored_regions = list(zip(region_scores, regions))
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            selected_cacheables, region_scores_by_chunk = (
                self._select_sliding_regions_with_scores(
                    scored_regions, final_token_budget=final_token_budget
                )
            )
            selected_by_doc = {}
            for chunk_idx, cacheable in selected_cacheables:
                selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
            for chunk_idx, cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], cacheables
                )
            summarized_batches.append(
                self._order_document_groups(summarized_docs, region_scores_by_chunk)
            )
        return summarized_batches


class DenseSlidingRegionMaxCompressor(_DenseSlidingRegionCompressor):
    """Score a region by its highest member-subchunk dense similarity."""

    region_score_aggregation = "max"


class DenseSlidingRegionAvgCompressor(_DenseSlidingRegionCompressor):
    """Score a region by its mean member-subchunk dense similarity."""

    region_score_aggregation = "avg"


class DenseSlidingSubchunkCompressor(_DenseSlidingRegionCompressor):
    """Directly embed and score each complete sliding subchunk at runtime."""

    region_score_aggregation = "direct_subchunk"

    @staticmethod
    def _direct_subchunk_cacheable(region) -> CacheableChunk:
        source_cacheables = region["source_cacheables"]
        selected_indices = [
            idx
            for idx in region["selected_indices"]
            if 0 <= idx < len(source_cacheables)
        ]
        selected = [source_cacheables[idx] for idx in selected_indices]
        return CacheableChunk(
            id=f"{region['region_id']}::direct_dense",
            text=" ".join(cacheable.text for cacheable in selected),
            parent_doc_id=region["parent_doc_id"],
            sentence_ids=[str(cacheable.id) for cacheable in selected],
            sentence_texts=[cacheable.text for cacheable in selected],
        )

    def _score_dense_regions_for_batch(self, batch_regions, batch_queries):
        batch_subchunks = [
            [self._direct_subchunk_cacheable(region) for region in regions]
            for regions in batch_regions
        ]
        flattened_subchunks = [
            subchunk for subchunks in batch_subchunks for subchunk in subchunks
        ]
        if not flattened_subchunks:
            return [[] for _ in batch_regions]

        query_embeddings = (
            self._embed_texts([self._format_query(query) for query in batch_queries])
            .detach()
            .to(torch.float32)
        )
        unique_embeddings, subchunk_index_by_key = self._embed_unique_cacheables(
            flattened_subchunks
        )
        unique_embeddings = unique_embeddings.to(query_embeddings.device)

        batch_scores = []
        for query_idx, subchunks in enumerate(batch_subchunks):
            if not subchunks:
                batch_scores.append([])
                continue
            row_indices = torch.tensor(
                [
                    subchunk_index_by_key[self._dense_cacheable_key(subchunk)]
                    for subchunk in subchunks
                ],
                dtype=torch.long,
                device=query_embeddings.device,
            )
            embeddings = unique_embeddings[row_indices]
            scores = torch.matmul(embeddings, query_embeddings[query_idx])
            batch_scores.append(
                [float(score) for score in scores.detach().cpu().tolist()]
            )
        return batch_scores
