"""Dense-embedding baseline for token-budgeted subchunk selection."""

import os
from typing import List

import torch

from chunk import RetrievableChunk
from compressor.base import Compressor
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

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        return self.embedder.embed_texts(texts).to(self.device)

    def _format_query(self, query: str) -> str:
        return f"{self.query_prefix}{query}"


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
