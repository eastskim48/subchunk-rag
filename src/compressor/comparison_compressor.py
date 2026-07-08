from typing import List
from chunk import RetrievableChunk, CacheableChunk
import re
from dotenv import load_dotenv
import os
from openai import OpenAI
import math
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from compressor.base import Compressor
from embedding_utils import BGE_M3_MODEL, DenseTextEmbedder, default_query_prefix
from materialize.colbert_window import (
    ColBERTWindowEncoder,
    default_colbert_repo_path,
    parse_bool,
)

QUERY_PREFIX = default_query_prefix(BGE_M3_MODEL)
# QUERY_PREFIX = "Represent this question for searching relevant passages: "


class FrontCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, document_text: str, query: str) -> str:
        del query
        sentences = re.split(r"(?<=[.!?])\s+", document_text.strip())
        front_sentences = [
            sentence.strip() for sentence in sentences if sentence.strip()
        ][:3]
        return " ".join(front_sentences)


class ComparisonSummarizer(Compressor):
    def __init__(self):
        super().__init__()
        print("Comparison summarization enabled. Initializing local embedding model...")
        self.embedding_model = os.getenv("COMPARE_EMBED_MODEL", BGE_M3_MODEL)
        self.device = os.getenv("COMPARE_EMBED_DEVICE", "cpu")
        self.batch_size = int(os.getenv("COMPARE_BATCH_SIZE", "128"))
        self.query_prefix = default_query_prefix(self.embedding_model)
        self.embedder = DenseTextEmbedder(
            model_name=self.embedding_model,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.device = self.embedder.device
        self.min_keep_ratio = 0.7
        self.max_keep_ratio = 0.9
        self.target_keep_ratio = 0.8

    def _score_chunk_texts_per_doc(self, docs: List[RetrievableChunk], query: str):
        chunk_texts_per_doc = []
        flattened_sentences = []
        for doc in docs:
            chunk_texts = [
                cacheable.text
                for cacheable in getattr(doc, "cacheables", [])
                if cacheable.text
            ]
            if chunk_texts:
                chunk_texts_per_doc.append(chunk_texts)
                flattened_sentences.extend(chunk_texts)
            else:
                chunk_texts_per_doc.append([])

        if not flattened_sentences:
            return chunk_texts_per_doc, None

        query_text = self._format_query(query)
        query_embedding = (
            self._embed_texts([query_text]).detach().cpu()[0].to(torch.float32)
        )
        sentence_embeddings = self._embed_texts_batched(flattened_sentences).to(
            torch.float32
        )
        similarities = torch.matmul(sentence_embeddings, query_embedding)
        return chunk_texts_per_doc, similarities

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc = []
        flattened_sentences = []
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
                    flattened_sentences.extend(chunk_texts)
                    sentence_owner_query_indices.extend([query_idx] * len(chunk_texts))
                else:
                    chunk_texts_per_doc.append([])
            batch_chunk_texts_per_doc.append(chunk_texts_per_doc)

        if not flattened_sentences:
            return batch_chunk_texts_per_doc, None

        query_texts = [self._format_query(query) for query in batch_queries]
        query_embeddings = (
            self._embed_texts(query_texts).detach().cpu().to(torch.float32)
        )
        sentence_embeddings = self._embed_texts_batched(flattened_sentences).to(
            torch.float32
        )
        owner_query_tensor = torch.tensor(
            sentence_owner_query_indices, dtype=torch.long
        )
        similarities = (sentence_embeddings * query_embeddings[owner_query_tensor]).sum(
            dim=1
        )
        return batch_chunk_texts_per_doc, similarities

    @staticmethod
    def _split_sentences(document_text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", document_text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        return self.embedder.embed_texts(texts).to(self.device)

    def _format_query(self, query: str) -> str:
        return f"{self.query_prefix}{query}"

    def _embed_texts_batched(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            hidden_size = self.embedder.embedding_dim
            return torch.empty((0, hidden_size), dtype=torch.float32)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch_embeddings = self._embed_texts(texts[start : start + self.batch_size])
            batches.append(batch_embeddings.detach().cpu())
        return (
            torch.cat(batches, dim=0)
            if batches
            else torch.empty((0, 0), dtype=torch.float32)
        )

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
        for docs, chunk_texts_per_doc in zip(
            batch_top_k_docs, batch_chunk_texts_per_doc
        ):
            summarized_docs = [doc.clone() for doc in docs]
            if batch_similarities is None:
                summarized_batches.append(summarized_docs)
                continue

            for doc_idx, chunk_texts in enumerate(chunk_texts_per_doc):
                if not chunk_texts:
                    continue
                doc_scores = batch_similarities[cursor : cursor + len(chunk_texts)]
                cursor += len(chunk_texts)
                # top_k = min(len(chunk_texts), max(3, int(len(chunk_texts) * 0.3)))
                top_k = min(3, len(chunk_texts))
                ranked_indices = torch.topk(doc_scores, k=top_k).indices.tolist()
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], ranked_indices
                )
            summarized_batches.append(summarized_docs)
        return summarized_batches


class GlobalComparisonSummarizer(ComparisonSummarizer):
    def __init__(self):
        super().__init__()
        self.global_top_r = self._parse_global_top_r(os.getenv("GLOBAL_TOP_R", "0.1"))

    @staticmethod
    def _parse_global_top_r(raw_value: str) -> float:
        value = raw_value.strip()
        if value.endswith("%"):
            value = value[:-1].strip()
            rate = float(value) / 100.0
        else:
            rate = float(value)
            if rate > 1.0:
                rate /= 100.0
        if rate <= 0:
            raise ValueError(f"GLOBAL_TOP_R must be positive, got {raw_value!r}")
        return rate

    def _global_top_count(self, sentence_count: int) -> int:
        if sentence_count <= 0:
            return 0
        return min(
            sentence_count, max(1, math.ceil(sentence_count * self.global_top_r))
        )

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
            GlobalComparisonSummarizer._deduplicate_sentence_refs_multi_scores(
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

            global_top_count = self._global_top_count(len(sentence_refs))
            if global_top_count <= 0:
                summarized_batches.append(summarized_chunks)
                continue

            top_indices = torch.topk(similarities, k=global_top_count).indices.tolist()
            selected_by_doc = {}
            for flat_idx in top_indices:
                chunk_idx, sentence_idx = sentence_refs[flat_idx]
                selected_by_doc.setdefault(chunk_idx, []).append(sentence_idx)

            for chunk_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[chunk_idx] = self._build_selected_document(
                    chunks[chunk_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)
        return summarized_batches


class MaterializedGlobalComparisonSummarizer(GlobalComparisonSummarizer):
    def __init__(self):
        super().__init__()
        self.compare_embed_dir = os.getenv("COMPARE_EMBED_DIR")
        if not self.compare_embed_dir:
            raise ValueError(
                "COMPARE_EMBED_DIR must be set for compress_method='compare_all_materialized'"
            )
        self.compare_embed_payload = self._load_compare_embed_payload()
        self.raw_embedding_pool = {}

    def _compare_embed_path(self) -> str:
        return os.path.join(self.compare_embed_dir, "compare_embed.pt")

    def _load_compare_embed_payload(self):
        payload_path = self._compare_embed_path()
        if not os.path.exists(payload_path):
            raise FileNotFoundError(
                f"materialized compare embedding file missing: {payload_path}"
            )
        payload = torch.load(payload_path, map_location="cpu")
        if payload.get("format") != "compare_embed_single_v1":
            raise ValueError(
                f"unsupported compare embed format: {payload.get('format')}"
            )
        payload_model = payload.get("embedding_model")
        if payload_model != self.embedding_model:
            raise ValueError(
                f"compare embed model mismatch: payload={payload_model!r} runtime={self.embedding_model!r}"
            )
        payload_query_prefix = payload.get("embedding_query_prefix", QUERY_PREFIX)
        if payload_query_prefix != self.query_prefix:
            raise ValueError(
                "compare embed query prefix mismatch: "
                f"payload={payload_query_prefix!r} runtime={self.query_prefix!r}"
            )

        return payload

    def _get_doc_index_entry(self, doc_id: str):
        return self.compare_embed_payload.get("doc_index", {}).get(doc_id)

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
                f"materialized compare embedding chunk_ids missing for {doc_id}: {missing_chunk_ids[:5]}"
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
            raise KeyError(f"materialized compare embedding index missing doc {doc.id}")

        row_start = int(doc_index_entry["row_start"])
        row_end = int(doc_index_entry["row_end"])
        embeddings = self.compare_embed_payload["embeddings"][row_start:row_end]
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"materialized compare embeddings for {doc.id} are not a tensor"
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


class MaterializedBudgetComparisonSummarizer(MaterializedGlobalComparisonSummarizer):
    def __init__(self):
        super().__init__()
        budget = os.getenv("COMPARE_FINAL_TOKEN_BUDGET") or os.getenv(
            "COLBERT_FINAL_TOKEN_BUDGET"
        )
        if not budget:
            raise ValueError(
                "compare_all_materialized_budget requires COMPARE_FINAL_TOKEN_BUDGET "
                "or COLBERT_FINAL_TOKEN_BUDGET"
            )
        self.final_token_budget = int(budget)
        if self.final_token_budget <= 0:
            raise ValueError(
                f"final token budget must be positive, got {self.final_token_budget}"
            )
        tokenizer_name = os.getenv(
            "COMPARE_BUDGET_TOKENIZER",
            os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0"),
        )
        self.budget_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(
            "Materialized dense comparison budget selection enabled. "
            f"budget={self.final_token_budget}, budget_tokenizer={tokenizer_name}"
        )

    def _token_count(self, text: str) -> int:
        return (
            len(
                self.budget_tokenizer(
                    text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                    verbose=False,
                )["input_ids"]
            )
            + 1
        )

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
            if not sentence_refs:
                summarized_batches.append(summarized_chunks)
                continue

            ranked_indices = torch.argsort(
                similarities, descending=True, stable=True
            ).tolist()
            selected_by_doc: dict[int, list[int]] = {}
            selected_ids = set()
            used_tokens = 0
            for flat_idx in ranked_indices:
                chunk_idx, sentence_idx = sentence_refs[flat_idx]
                cacheables = getattr(chunks[chunk_idx], "cacheables", []) or []
                if not (0 <= sentence_idx < len(cacheables)):
                    continue
                cacheable = cacheables[sentence_idx]
                if cacheable.id in selected_ids:
                    continue
                token_len = self._token_count(cacheable.text)
                if selected_ids and used_tokens + token_len > self.final_token_budget:
                    continue
                selected_ids.add(cacheable.id)
                selected_by_doc.setdefault(chunk_idx, []).append(sentence_idx)
                used_tokens += token_len
                if used_tokens >= self.final_token_budget:
                    break

            for chunk_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[chunk_idx] = self._build_selected_document(
                    chunks[chunk_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)
        return summarized_batches


class DenseSlidingRegionSummarizer(ComparisonSummarizer):
    def __init__(self):
        super().__init__()
        final_budget = os.getenv("COMPARE_FINAL_TOKEN_BUDGET") or os.getenv(
            "COLBERT_FINAL_TOKEN_BUDGET"
        )
        if not final_budget:
            raise ValueError(
                "dense_sliding_region requires COMPARE_FINAL_TOKEN_BUDGET or COLBERT_FINAL_TOKEN_BUDGET"
            )
        self.final_token_budget = int(final_budget)
        if self.final_token_budget <= 0:
            raise ValueError(
                f"final token budget must be positive, got {self.final_token_budget}"
            )
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()
        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        self.window_encoder = ColBERTWindowEncoder(
            model_name=model_name,
            repo_path=repo_path,
            device="cpu",
            batch_size=1,
            max_length=int(os.getenv("COLBERT_SLIDING_WINDOW_TOKEN_BUDGET", "180")),
            disable_cpu_extension=parse_bool(
                os.getenv("COLBERT_DISABLE_CPU_EXTENSION", "True")
            ),
            verify_tensorization=False,
        )
        self.region_token_budget = self.window_encoder.max_length
        print(
            "Dense sliding-region compression enabled. "
            f"region_token_budget={self.region_token_budget}, final_token_budget={self.final_token_budget}"
        )

    @staticmethod
    def _build_unselected_document(doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned

    @staticmethod
    def _build_region_document(
        doc: RetrievableChunk, selected_cacheables: list[CacheableChunk]
    ) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = [cacheable.clone() for cacheable in selected_cacheables]
        return cloned

    def _sliding_regions_for_doc(self, doc: RetrievableChunk, chunk_idx: int):
        cacheables = [
            cacheable
            for cacheable in getattr(doc, "cacheables", []) or []
            if cacheable.text
        ]
        if not cacheables:
            return []
        specs = self.window_encoder.build_centered_windows(
            sentences=[cacheable.text for cacheable in cacheables],
            token_budget=self.region_token_budget,
        )
        regions = []
        seen_region_keys = set()
        for center_idx, spec in enumerate(specs):
            selected_indices = tuple(
                idx for idx in spec.selected_indices if 0 <= idx < len(cacheables)
            )
            if not selected_indices or selected_indices in seen_region_keys:
                continue
            seen_region_keys.add(selected_indices)
            sentence_ids = [cacheables[idx].id for idx in selected_indices]
            cacheable = CacheableChunk(
                id=f"{doc.id}::dense_sliding_region_{center_idx}",
                text=spec.text,
                parent_doc_id=doc.id,
                chunk_size=self.region_token_budget,
                sentence_ids=sentence_ids,
                sentence_texts=[cacheables[idx].text for idx in selected_indices],
            )
            regions.append(
                {
                    "chunk_idx": chunk_idx,
                    "cacheable": cacheable,
                    "selected_indices": selected_indices,
                    "source_cacheables": cacheables,
                }
            )
        return regions

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        all_region_texts: list[str] = []
        batch_regions = []
        for docs in batch_top_k_docs:
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx))
            batch_regions.append(regions)
            all_region_texts.extend(region["cacheable"].text for region in regions)

        if not all_region_texts:
            return [
                [self._build_unselected_document(doc) for doc in docs]
                for docs in batch_top_k_docs
            ]

        region_embeddings = self._embed_texts_batched(all_region_texts).to(
            torch.float32
        )
        query_embeddings = (
            self._embed_texts([self._format_query(query) for query in batch_queries])
            .detach()
            .cpu()
            .to(torch.float32)
        )

        cursor = 0
        for query_idx, (docs, regions) in enumerate(
            zip(batch_top_k_docs, batch_regions)
        ):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            local_embeddings = region_embeddings[cursor : cursor + len(regions)]
            cursor += len(regions)
            scores = torch.matmul(local_embeddings, query_embeddings[query_idx])
            ranked_indices = torch.argsort(
                scores, descending=True, stable=True
            ).tolist()

            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            selected_sentence_ids = set()
            used_tokens = 0
            for region_idx in ranked_indices:
                region = regions[region_idx]
                novel_sentence_cacheables = []
                for idx in region["selected_indices"]:
                    source = region["source_cacheables"][idx]
                    if source.id in selected_sentence_ids:
                        continue
                    token_len = self.window_encoder.token_count(source.text)
                    if (
                        novel_sentence_cacheables
                        and used_tokens + token_len > self.final_token_budget
                    ):
                        continue
                    if (
                        not novel_sentence_cacheables
                        and selected_sentence_ids
                        and used_tokens + token_len > self.final_token_budget
                    ):
                        continue
                    novel_sentence_cacheables.append(source)
                    selected_sentence_ids.add(source.id)
                    used_tokens += token_len
                    if used_tokens >= self.final_token_budget:
                        break
                if novel_sentence_cacheables:
                    first = novel_sentence_cacheables[0]
                    region_cacheable = CacheableChunk(
                        id=f"{region['cacheable'].id}::dedup",
                        text=" ".join(
                            cacheable.text for cacheable in novel_sentence_cacheables
                        ),
                        parent_doc_id=region["cacheable"].parent_doc_id,
                        chunk_size=self.region_token_budget,
                        sentence_ids=[
                            cacheable.id for cacheable in novel_sentence_cacheables
                        ],
                        sentence_texts=[
                            cacheable.text for cacheable in novel_sentence_cacheables
                        ],
                        chunk_start=first.chunk_start,
                        chunk_end=novel_sentence_cacheables[-1].chunk_end,
                    )
                    selected_by_doc.setdefault(region["chunk_idx"], []).append(
                        region_cacheable
                    )
                if used_tokens >= self.final_token_budget:
                    break

            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


def _reciprocal_rank_fusion_by_owner(
    raw_scores: torch.Tensor,
    title_scores: torch.Tensor,
    owner_indices: List[int],
    rrf_k: float,
) -> torch.Tensor:
    if raw_scores.shape != title_scores.shape:
        raise ValueError(
            f"title RRF score shape mismatch: raw={tuple(raw_scores.shape)} title={tuple(title_scores.shape)}"
        )
    if raw_scores.numel() == 0:
        return raw_scores.to(torch.float32)
    if rrf_k < 0:
        raise ValueError(f"TITLE_RRF_K must be non-negative, got {rrf_k}")

    owner_tensor = torch.tensor(owner_indices, dtype=torch.long)
    if owner_tensor.numel() != raw_scores.numel():
        raise ValueError(
            f"title RRF owner length mismatch: owners={owner_tensor.numel()} scores={raw_scores.numel()}"
        )

    fused = torch.zeros_like(raw_scores, dtype=torch.float32)
    for owner_idx in owner_tensor.unique().tolist():
        mask = owner_tensor == owner_idx
        local_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if local_indices.numel() == 0:
            continue

        raw_order = torch.argsort(
            raw_scores[mask].to(torch.float32), descending=True, stable=True
        )
        title_order = torch.argsort(
            title_scores[mask].to(torch.float32), descending=True, stable=True
        )
        raw_ranks = torch.empty(local_indices.numel(), dtype=torch.float32)
        title_ranks = torch.empty(local_indices.numel(), dtype=torch.float32)
        raw_ranks[raw_order] = torch.arange(
            1, local_indices.numel() + 1, dtype=torch.float32
        )
        title_ranks[title_order] = torch.arange(
            1, local_indices.numel() + 1, dtype=torch.float32
        )

        fused[local_indices] = 1.0 / (rrf_k + raw_ranks) + 1.0 / (rrf_k + title_ranks)
    return fused


class TitleRRFSummarizer(MaterializedGlobalComparisonSummarizer):
    def __init__(self):
        super().__init__()
        self.title_compare_embed_dir = self._resolve_title_compare_embed_dir()
        self.title_compare_embed_payload = self._load_compare_embed_payload_from_dir(
            self.title_compare_embed_dir
        )
        self._validate_title_compare_embed_payload()
        self.title_embedding_pool = {}
        self.rrf_k = float(os.getenv("TITLE_RRF_K", os.getenv("HYBRID_RRF_K", "60")))
        if self.rrf_k < 0:
            raise ValueError(f"TITLE_RRF_K must be non-negative, got {self.rrf_k}")
        print(
            "Title RRF subchunk selection enabled. "
            f"raw_compare_embed_dir={self.compare_embed_dir} "
            f"title_compare_embed_dir={self.title_compare_embed_dir} "
            f"TITLE_RRF_K={self.rrf_k}"
        )

    @staticmethod
    def _resolve_title_compare_embed_dir() -> str:
        explicit_dir = os.getenv("TITLE_COMPARE_EMBED_DIR")
        if explicit_dir:
            return explicit_dir

        dataset_path = os.getenv("DATASET_PATH")
        if dataset_path:
            return os.path.join(dataset_path, "sent-title-test", "compare_embed")

        raise ValueError(
            "TITLE_COMPARE_EMBED_DIR must be set for compress_method='title_rrf', "
            "or DATASET_PATH must point to a dataset containing sent-title-test/compare_embed"
        )

    @staticmethod
    def _load_compare_embed_payload_from_dir(compare_embed_dir: str):
        payload_path = os.path.join(compare_embed_dir, "compare_embed.pt")
        if not os.path.exists(payload_path):
            raise FileNotFoundError(
                f"title compare embedding file missing: {payload_path}"
            )
        payload = torch.load(payload_path, map_location="cpu")
        if payload.get("format") != "compare_embed_single_v1":
            raise ValueError(
                f"unsupported title compare embed format: {payload.get('format')}"
            )
        return payload

    def _validate_title_compare_embed_payload(self) -> None:
        payload_model = self.title_compare_embed_payload.get("embedding_model")
        if payload_model != self.embedding_model:
            raise ValueError(
                f"title compare embed model mismatch: payload={payload_model!r} runtime={self.embedding_model!r}"
            )
        payload_query_prefix = self.title_compare_embed_payload.get(
            "embedding_query_prefix", QUERY_PREFIX
        )
        if payload_query_prefix != self.query_prefix:
            raise ValueError(
                "title compare embed query prefix mismatch: "
                f"payload={payload_query_prefix!r} runtime={self.query_prefix!r}"
            )

    @staticmethod
    def _owner_indices_from_batch_chunk_texts(batch_chunk_texts_per_doc) -> List[int]:
        owner_indices = []
        for query_idx, chunk_texts_per_doc in enumerate(batch_chunk_texts_per_doc):
            count = sum(len(chunk_texts) for chunk_texts in chunk_texts_per_doc)
            owner_indices.extend([query_idx] * count)
        return owner_indices

    def _get_title_doc_index_entry(self, doc_id: str):
        return self.title_compare_embed_payload.get("doc_index", {}).get(doc_id)

    def _load_title_doc_embeddings(self, doc: RetrievableChunk) -> torch.Tensor:
        if doc.id in self.title_embedding_pool:
            return self.title_embedding_pool[doc.id]

        expected_chunk_ids = [
            cacheable.id for cacheable in getattr(doc, "cacheables", [])
        ]
        doc_index_entry = self._get_title_doc_index_entry(doc.id)
        if doc_index_entry is None:
            source_doc_id = doc.metadata.get("parent_doc_id", doc.id)
            doc_index_entry = self._get_title_doc_index_entry(source_doc_id)
        if doc_index_entry is None:
            raise KeyError(f"title compare embedding index missing doc {doc.id}")

        row_start = int(doc_index_entry["row_start"])
        row_end = int(doc_index_entry["row_end"])
        embeddings = self.title_compare_embed_payload["embeddings"][row_start:row_end]
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"title compare embeddings for {doc.id} are not a tensor")
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
        self.title_embedding_pool[doc.id] = embeddings
        return embeddings

    def _title_scores_for_batch(
        self,
        batch_top_k_docs: List[List[RetrievableChunk]],
        batch_queries: List[str],
    ) -> torch.Tensor:
        flattened_embeddings = []
        owner_indices = []

        for query_idx, docs in enumerate(batch_top_k_docs):
            for doc in docs:
                cacheables = [
                    cacheable
                    for cacheable in getattr(doc, "cacheables", [])
                    if cacheable.text
                ]
                if not cacheables:
                    continue
                flattened_embeddings.append(self._load_title_doc_embeddings(doc))
                owner_indices.extend([query_idx] * len(cacheables))

        if not flattened_embeddings:
            return torch.empty(0, dtype=torch.float32)

        query_texts = [self._format_query(query) for query in batch_queries]
        query_embeddings = (
            self._embed_texts(query_texts).detach().cpu().to(torch.float32)
        )
        sentence_embeddings = torch.cat(flattened_embeddings, dim=0)
        owner_query_tensor = torch.tensor(owner_indices, dtype=torch.long)
        return (sentence_embeddings * query_embeddings[owner_query_tensor]).sum(dim=1)

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, raw_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        if raw_scores is None:
            return batch_chunk_texts_per_doc, None

        title_scores = self._title_scores_for_batch(
            batch_top_k_docs=batch_top_k_docs,
            batch_queries=batch_queries,
        )
        if title_scores.shape != raw_scores.shape:
            raise ValueError(
                f"title RRF score shape mismatch: raw={tuple(raw_scores.shape)} title={tuple(title_scores.shape)}"
            )

        owner_indices = self._owner_indices_from_batch_chunk_texts(
            batch_chunk_texts_per_doc
        )
        rrf_scores = _reciprocal_rank_fusion_by_owner(
            raw_scores=raw_scores.to(torch.float32),
            title_scores=title_scores.to(torch.float32),
            owner_indices=owner_indices,
            rrf_k=self.rrf_k,
        )
        return batch_chunk_texts_per_doc, rrf_scores

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, raw_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        summarized_batches = []
        raw_cursor = 0
        if raw_scores is None:
            return [
                [self._build_unselected_document(chunk) for chunk in chunks]
                for chunks in batch_top_k_docs
            ]

        title_scores = self._title_scores_for_batch(
            batch_top_k_docs=batch_top_k_docs,
            batch_queries=batch_queries,
        )
        if title_scores.shape != raw_scores.shape:
            raise ValueError(
                f"title RRF score shape mismatch: raw={tuple(raw_scores.shape)} title={tuple(title_scores.shape)}"
            )

        for chunks, chunk_texts_per_doc in zip(
            batch_top_k_docs, batch_chunk_texts_per_doc
        ):
            summarized_chunks = [
                self._build_unselected_document(chunk) for chunk in chunks
            ]

            sentence_refs = []
            for chunk_idx, chunk_texts in enumerate(chunk_texts_per_doc):
                for sentence_idx in range(len(chunk_texts)):
                    sentence_refs.append((chunk_idx, sentence_idx))

            local_count = len(sentence_refs)
            local_raw_scores = raw_scores[raw_cursor : raw_cursor + local_count]
            local_title_scores = title_scores[raw_cursor : raw_cursor + local_count]
            raw_cursor += local_count

            sentence_refs, dedup_scores = self._deduplicate_sentence_refs_multi_scores(
                chunks=chunks,
                sentence_refs=sentence_refs,
                score_tensors=[local_raw_scores, local_title_scores],
            )
            if not sentence_refs:
                summarized_batches.append(summarized_chunks)
                continue

            local_raw_scores, local_title_scores = dedup_scores
            rrf_scores = _reciprocal_rank_fusion_by_owner(
                raw_scores=local_raw_scores.to(torch.float32),
                title_scores=local_title_scores.to(torch.float32),
                owner_indices=[0] * len(sentence_refs),
                rrf_k=self.rrf_k,
            )

            global_top_count = self._global_top_count(len(sentence_refs))
            if global_top_count <= 0:
                summarized_batches.append(summarized_chunks)
                continue

            top_indices = torch.topk(rrf_scores, k=global_top_count).indices.tolist()
            selected_by_doc = {}
            for flat_idx in top_indices:
                chunk_idx, sentence_idx = sentence_refs[flat_idx]
                selected_by_doc.setdefault(chunk_idx, []).append(sentence_idx)

            for chunk_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[chunk_idx] = self._build_selected_document(
                    chunks[chunk_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)

        return summarized_batches


class WindowedComparisonSummarizer(ComparisonSummarizer):
    def __init__(self):
        super().__init__()
        self.window_size = int(os.getenv("COMPARE_WINDOW_SIZE", "1"))

    def _expand_with_local_window(
        self, selected_indices: List[int], total_chunks: int
    ) -> List[int]:
        if total_chunks <= 0:
            return []

        expanded = set()
        for idx in selected_indices:
            start = max(0, idx - self.window_size)
            end = min(total_chunks - 1, idx + self.window_size)
            expanded.update(range(start, end + 1))
        return sorted(expanded)

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        base_batches = super().compress_batch_top_k_docs(
            batch_top_k_docs, batch_queries
        )
        expanded_batches = []
        for original_docs, selected_docs in zip(batch_top_k_docs, base_batches):
            expanded_docs = []
            for original_doc, selected_doc in zip(original_docs, selected_docs):
                original_cacheables = list(
                    getattr(original_doc, "cacheables", []) or []
                )
                selected_cacheables = list(
                    getattr(selected_doc, "cacheables", []) or []
                )
                if not original_cacheables or not selected_cacheables:
                    expanded_docs.append(selected_doc)
                    continue

                index_by_chunk_id = {
                    cacheable.id: idx
                    for idx, cacheable in enumerate(original_cacheables)
                }
                selected_indices = [
                    index_by_chunk_id[cacheable.id]
                    for cacheable in selected_cacheables
                    if cacheable.id in index_by_chunk_id
                ]
                expanded_indices = self._expand_with_local_window(
                    selected_indices, len(original_cacheables)
                )
                expanded_docs.append(
                    self._build_selected_document(original_doc, expanded_indices)
                )
            expanded_batches.append(expanded_docs)
        return expanded_batches


class Summarizer(Compressor):
    def __init__(self):
        super().__init__()
        load_dotenv()
        print("Summarization enabled. Initializing OpenAI client...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env, but summarize=True was requested."
            )
        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(**client_kwargs)
        self.keep_ratio_min = 0.7
        self.keep_ratio_max = 0.9

    def _length_budget(self, document_text: str) -> tuple[int, int, int]:
        words = document_text.split()
        original_words = max(len(words), 1)
        min_words = max(20, math.ceil(original_words * self.keep_ratio_min))
        max_words = max(min_words, math.ceil(original_words * self.keep_ratio_max))
        return original_words, min_words, max_words

    def _build_batch_prompt(self, rchunks: List[RetrievableChunk], query: str) -> str:
        parts = []
        for idx, rchunk in enumerate(rchunks):
            original_words, min_words, max_words = self._length_budget(rchunk.text)
            parts.append(
                f"[DOC {idx}]\n"
                f"id: {rchunk.id}\n"
                f"original_length_words: {original_words}\n"
                f"target_summary_words: {min_words}-{max_words}\n"
                f"text:\n{rchunk.text}"
            )
        docs_block = "\n\n".join(parts)
        return (
            f"Summarize each document separately for answering question: {query}\n"
            "Keep each summary factual and grounded in the source.\n"
            "Do not add unsupported information.\n"
            "Do not compress too aggressively: preserve most of the useful evidence, "
            "including names, dates, entities, comparisons, and qualifiers that may affect the answer.\n"
            f"For each document, keep roughly {self.keep_ratio_min * 100} to {self.keep_ratio_max * 100} of the original length.\n"
            "Remove only clearly irrelevant detail. If unsure, keep the detail rather than dropping it.\n\n"
            "Return valid JSON only in this format:\n"
            '{"summaries": [{"index": 0, "summary": "..."}]}\n\n'
            f"Documents:\n{docs_block}"
        )

    def _parse_batch_response(
        self, content: str, retrievable_chunks: List[RetrievableChunk]
    ) -> List[RetrievableChunk]:
        try:
            parsed = json.loads(content)
            items = parsed.get("summaries", [])
            by_index = {
                int(item["index"]): str(item.get("summary", "")).strip()
                for item in items
                if "index" in item
            }
        except Exception:
            by_index = {}

        summarized_docs = []
        for idx, rchunk in enumerate(retrievable_chunks):
            summary = by_index.get(idx, rchunk.text)
            cloned = rchunk.clone()
            cloned.text = summary or rchunk.text
            summarized_docs.append(cloned)
        return summarized_docs

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize retrieved documents for question answering. "
                            "Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_batch_prompt(docs, query),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            summarized_batches.append(self._parse_batch_response(content, docs))
        return summarized_batches

    def compress(self, document_text: str, query: str) -> str:
        original_words, min_words, max_words = self._length_budget(document_text)
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize retrieved documents for question answering. "
                        "Keep the summary factual and grounded in the source. "
                        "Do not add unsupported information. "
                        "Do not compress too aggressively: preserve most of the useful evidence, "
                        "including names, dates, entities, comparisons, and qualifiers that may affect the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Document:\n{document_text}\n\n"
                        f"Original length: about {original_words} words.\n"
                        f"Write a concise summary that keeps roughly 50% to 70% of the original length "
                        f"(target: {min_words} to {max_words} words).\n"
                        "Remove only clearly irrelevant detail. If unsure, keep the detail rather than dropping it.\n"
                        "Return only the summary."
                    ),
                },
            ],
            temperature=0,
        )
        summary = response.choices[0].message.content
        return summary.strip() if summary else ""
