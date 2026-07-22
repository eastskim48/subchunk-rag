import math
import os
import time
import json
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer

from chunk import CacheableChunk, RetrievableChunk
from compressor.base import Compressor
from compressor.comparison_compressor import parse_global_top_r
from materialize.colbert_window import (
    ColBERTWindowArtifact,
    ColBERTWindowEncoder,
    default_colbert_repo_path,
    global_top_count,
    score_maxsim,
)
from materialize.db_manifest import read_db_build_manifest


def _configured_db_dir() -> str | None:
    db_dir = os.getenv("DB_DIR")
    if db_dir:
        return db_dir
    dataset_path = os.getenv("DATASET_PATH")
    data_subdir = os.getenv("DATA_SUBDIR")
    if dataset_path and data_subdir:
        return os.path.join(dataset_path, data_subdir, "db")
    return None


def _resolve_configured_retrieval_chunk_size() -> int:
    db_dir = _configured_db_dir()
    if db_dir is None:
        raise ValueError(
            "DB_DIR or DATASET_PATH/DATA_SUBDIR is required for retrieval chunk "
            "validation"
        )
    manifest = read_db_build_manifest(db_dir)
    if manifest is None:
        raise ValueError(f"DB build manifest is required for ColBERT runtime: {db_dir}")
    manifest_size = manifest.get("retrievable_chunk_size")
    if isinstance(manifest_size, bool) or not isinstance(manifest_size, int):
        raise ValueError(
            "DB build manifest retrievable_chunk_size must be an integer, "
            f"got {manifest_size!r}"
        )
    if manifest_size <= 0:
        raise ValueError(
            "DB build manifest retrievable_chunk_size must be positive, "
            f"got {manifest_size}"
        )
    return manifest_size


def _resolve_budget_tokenizer_name() -> str:
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise ValueError(
            "MODEL_NAME must be set for ColBERT final budget accounting; "
            "the final prompt budget must use the evaluated LLM tokenizer"
        )
    return model_name


def _configured_colbert_rerank_keep() -> int:
    keep = int(os.getenv("COLBERT_RERANK_KEEP", "0"))
    if keep <= 0:
        raise ValueError("COLBERT_RERANK_KEEP must be positive")
    return keep


class FixedChunkColBERTRerankArtifact:
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


class FixedChunkColBERTRerankSummarizer(Compressor):
    def __init__(self):
        super().__init__()
        dataset_path = os.getenv("DATASET_PATH")
        data_subdir = os.getenv("DATA_SUBDIR")
        if not dataset_path or not data_subdir:
            raise ValueError("DATASET_PATH and DATA_SUBDIR are required")
        artifact_dir = os.path.join(
            dataset_path, data_subdir, "colbert_fixed_chunk_docmax512"
        )
        self.keep = int(os.getenv("COLBERT_CHUNK_RERANK_KEEP", "0"))
        if self.keep <= 0:
            raise ValueError("COLBERT_CHUNK_RERANK_KEEP must be positive")
        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()
        batch_size = int(os.getenv("COLBERT_BATCH_SIZE", "32"))

        print(f"Fixed-chunk ColBERT rerank enabled. Loading artifact: {artifact_dir}")
        self.artifact = FixedChunkColBERTRerankArtifact(artifact_dir)
        artifact_model_name = self.artifact.index.get("model_name")
        if artifact_model_name != model_name:
            raise ValueError(
                "COLBERT_MODEL_NAME does not match the fixed-chunk ColBERT artifact: "
                f"runtime={model_name!r}, artifact={artifact_model_name!r}"
            )
        self.encoder = ColBERTWindowEncoder(
            model_name=model_name,
            repo_path=repo_path,
            device="cpu",
            batch_size=batch_size,
            max_length=int(self.artifact.index.get("official_doc_maxlen", 0)),
            doc_maxlen=int(self.artifact.index.get("official_doc_maxlen", 0)),
            query_maxlen=None,
            disable_cpu_extension=True,
            verify_tensorization=False,
        )
        self.query_encoder_warmup_time = 0.0
        self.last_profile: dict[str, float | int] = {}

    def warmup_query_encoder(self) -> float:
        start = time.perf_counter()
        self.encoder.encode_queries(["warmup query"])
        self.query_encoder_warmup_time = time.perf_counter() - start
        return self.query_encoder_warmup_time

    def clear_inter_batch_cache(self) -> None:
        return None

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = {
            "query_encode_time": 0.0,
            "artifact_lookup_time": 0.0,
            "score_time": 0.0,
            "sort_time": 0.0,
            "build_output_time": 0.0,
            "query_count": len(batch_queries),
            "retrieved_doc_count": sum(len(docs) for docs in batch_top_k_docs),
            "selected_doc_count": 0,
        }
        start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - start

        summarized_batches = []
        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            scored_docs = []
            for doc in docs:
                lookup_start = time.perf_counter()
                vectors = self.artifact.vectors_for_chunk_id(str(doc.id))
                profile["artifact_lookup_time"] += time.perf_counter() - lookup_start
                score_start = time.perf_counter()
                score = score_maxsim(query_vector, vectors)
                profile["score_time"] += time.perf_counter() - score_start
                scored_docs.append((score, doc))

            sort_start = time.perf_counter()
            scored_docs.sort(key=lambda item: item[0], reverse=True)
            selected = scored_docs[: min(self.keep, len(scored_docs))]
            profile["sort_time"] += time.perf_counter() - sort_start

            build_start = time.perf_counter()
            summarized_batches.append([doc.clone() for _, doc in selected])
            profile["selected_doc_count"] += len(selected)
            profile["build_output_time"] += time.perf_counter() - build_start

        self.last_profile = profile
        return summarized_batches


class ColBERTWindowSummarizer(Compressor):
    def __init__(
        self,
        *,
        initialize_global_top_r: bool | None = None,
        initialize_token_budget: bool | None = None,
    ):
        super().__init__()
        retain_token_ratio = os.getenv("RETAIN_TOKEN_RATIO")
        final_token_budget = os.getenv("COLBERT_FINAL_TOKEN_BUDGET")
        if initialize_token_budget is None:
            initialize_token_budget = (
                retain_token_ratio is not None or final_token_budget is not None
            )
        if initialize_global_top_r is None:
            initialize_global_top_r = not initialize_token_budget
        if initialize_global_top_r:
            self.global_top_r = parse_global_top_r(os.getenv("GLOBAL_TOP_R", "0.1"))
        if initialize_token_budget:
            self.retain_token_ratio = (
                self._parse_retain_token_ratio(retain_token_ratio)
                if retain_token_ratio is not None
                else None
            )
            self.final_token_budget = (
                int(final_token_budget) if final_token_budget else None
            )
            if self.final_token_budget is not None and self.final_token_budget <= 0:
                raise ValueError(
                    "COLBERT_FINAL_TOKEN_BUDGET must be positive, got "
                    f"{self.final_token_budget}"
                )
            self._token_len_cache: dict[str, int] = {}
        self.last_profile: dict[str, float | int] = {}
        artifact_dir = os.getenv("COLBERT_WINDOW_DIR")
        if not artifact_dir:
            dataset_path = os.getenv("DATASET_PATH")
            data_subdir = os.getenv("DATA_SUBDIR", "sent")
            if not dataset_path:
                raise ValueError(
                    "COLBERT_WINDOW_DIR or DATASET_PATH must be set for colbert_subchunk compression"
                )
            artifact_dir = os.path.join(dataset_path, data_subdir, "colbert_window")

        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        batch_size = int(os.getenv("COLBERT_BATCH_SIZE", "32"))
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()

        print(f"ColBERT window compression enabled. Loading artifact: {artifact_dir}")
        self.artifact = ColBERTWindowArtifact(artifact_dir)
        if getattr(self.artifact, "db_manifest_reference", None) is not None:
            db_dir = _configured_db_dir()
            if db_dir is None:
                raise ValueError(
                    "DB_DIR or DATASET_PATH/DATA_SUBDIR is required to validate "
                    "the ColBERT artifact DB manifest"
                )
            self.artifact.validate_db_manifest(db_dir)
        if initialize_token_budget:
            budget_tokenizer_name = _resolve_budget_tokenizer_name()
            self.budget_tokenizer_name = budget_tokenizer_name
            self.budget_tokenizer = AutoTokenizer.from_pretrained(budget_tokenizer_name)
            print(
                "ColBERT final budget tokenizer enabled: " f"{budget_tokenizer_name!r}"
            )
        artifact_model_name = self.artifact.index.get(
            "checkpoint_name"
        ) or self.artifact.index.get("model_name")
        if artifact_model_name != model_name:
            raise ValueError(
                "COLBERT_MODEL_NAME does not match the ColBERT window artifact: "
                f"runtime={model_name!r}, artifact={artifact_model_name!r}"
            )
        self.encoder = ColBERTWindowEncoder(
            model_name=model_name,
            repo_path=repo_path,
            device="cpu",
            batch_size=batch_size,
            max_length=int(self.artifact.index.get("official_doc_maxlen", 0)),
            query_maxlen=int(self.artifact.index["official_query_maxlen"]),
            disable_cpu_extension=True,
            verify_tensorization=False,
        )
        self.query_encoder_warmup_time = 0.0

    def warmup_query_encoder(self) -> float:
        start = time.perf_counter()
        self.encoder.encode_queries(["warmup query"])
        self.query_encoder_warmup_time = time.perf_counter() - start
        return self.query_encoder_warmup_time

    def clear_inter_batch_cache(self) -> None:
        self.artifact.retrievable_vectors_cache.clear()

    @staticmethod
    def _empty_profile() -> dict[str, float | int]:
        return {
            "query_encode_time": 0.0,
            "budget_time": 0.0,
            "artifact_lookup_time": 0.0,
            "region_spec_time": 0.0,
            "region_object_time": 0.0,
            "sentence_maxsim_time": 0.0,
            "region_score_time": 0.0,
            "sort_time": 0.0,
            "select_time": 0.0,
            "build_output_time": 0.0,
            "query_count": 0,
            "retrieved_doc_count": 0,
            "region_count": 0,
            "unique_sentence_count": 0,
            "sentence_token_count": 0,
        }

    @staticmethod
    def _profile_add(profile: dict[str, float | int] | None, key: str, value):
        if profile is not None:
            profile[key] = profile.get(key, 0) + value

    @staticmethod
    def _parse_retain_token_ratio(raw_value: str) -> float:
        value = raw_value.strip()
        if value.endswith("%"):
            value = value[:-1].strip()
            ratio = float(value) / 100.0
        else:
            ratio = float(value)
            if ratio > 1.0:
                ratio /= 100.0
        if ratio <= 0:
            raise ValueError(f"RETAIN_TOKEN_RATIO must be positive, got {raw_value!r}")
        return ratio

    def _retrieved_context_token_count(self, docs: List[RetrievableChunk]) -> int:
        seen_ids = set()
        unique_cacheables = []
        for doc in docs:
            for cacheable in getattr(doc, "cacheables", []) or []:
                text = getattr(cacheable, "text", None)
                if not text:
                    continue
                cacheable_id = getattr(cacheable, "id", None)
                if cacheable_id:
                    if cacheable_id in seen_ids:
                        continue
                    seen_ids.add(cacheable_id)
                unique_cacheables.append(cacheable)
        return sum(self._cacheable_token_lens(unique_cacheables))

    def _resolve_final_token_budget(
        self, docs: List[RetrievableChunk], absolute_budget: int | None
    ) -> int | None:
        retain_token_ratio = getattr(self, "retain_token_ratio", None)
        if retain_token_ratio is None:
            return absolute_budget
        retrieved_tokens = self._retrieved_context_token_count(docs)
        if retrieved_tokens <= 0:
            return 0
        return max(1, math.ceil(retrieved_tokens * retain_token_ratio))

    @staticmethod
    def _build_unselected_document(doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned

    def _stored_prompt_token_count(self, cacheable) -> int | None:
        prompt_token_count = getattr(cacheable, "prompt_token_count", None)
        if prompt_token_count is None:
            return None
        prompt_tokenizer_name = getattr(cacheable, "prompt_tokenizer_name", None)
        if prompt_tokenizer_name != self.budget_tokenizer_name:
            return None
        return int(prompt_token_count)

    def _cacheable_token_len(self, cacheable) -> int:
        cacheable_id = getattr(cacheable, "id", None)
        if cacheable_id:
            cached = self._token_len_cache.get(cacheable_id)
            if cached is not None:
                return cached
        stored_token_count = self._stored_prompt_token_count(cacheable)
        if stored_token_count is not None:
            if cacheable_id:
                self._token_len_cache[cacheable_id] = stored_token_count
            return stored_token_count
        token_len = self._prompt_visible_token_count(cacheable)
        if cacheable_id:
            self._token_len_cache[cacheable_id] = token_len
        return token_len

    def _cacheable_token_lens(self, cacheables) -> list[int]:
        lengths: list[int | None] = []
        missing = []
        missing_positions = []
        for position, cacheable in enumerate(cacheables):
            cacheable_id = getattr(cacheable, "id", None)
            cached = self._token_len_cache.get(cacheable_id) if cacheable_id else None
            if cached is None:
                stored_token_count = self._stored_prompt_token_count(cacheable)
                if stored_token_count is not None:
                    if cacheable_id:
                        self._token_len_cache[cacheable_id] = stored_token_count
                    lengths.append(stored_token_count)
                    continue
                lengths.append(None)
                missing.append(cacheable)
                missing_positions.append(position)
            else:
                lengths.append(cached)

        if missing:
            token_lengths = self._prompt_visible_token_counts(missing)
            for cacheable, position, token_len in zip(
                missing, missing_positions, token_lengths
            ):
                lengths[position] = token_len
                cacheable_id = getattr(cacheable, "id", None)
                if cacheable_id:
                    self._token_len_cache[cacheable_id] = token_len

        return [int(length) for length in lengths if length is not None]

    def _format_budget_cacheable_text(self, cacheable) -> str:
        return f"{cacheable.text.strip()}\n\n"

    def _prompt_visible_token_count(self, cacheable) -> int:
        return self._prompt_visible_token_counts([cacheable])[0]

    def _prompt_visible_token_counts(self, cacheables) -> list[int]:
        texts = [
            self._format_budget_cacheable_text(cacheable) for cacheable in cacheables
        ]
        if not texts:
            return []
        encoded = self.budget_tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            verbose=False,
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    @staticmethod
    def _unique_cacheables(cacheables) -> list:
        unique = []
        seen_ids = set()
        for cacheable in cacheables:
            cacheable_id = getattr(cacheable, "id", None)
            if cacheable_id:
                if cacheable_id in seen_ids:
                    continue
                seen_ids.add(cacheable_id)
            unique.append(cacheable)
        return unique

    def _iter_candidates(self, docs: List[RetrievableChunk]):
        candidates = []
        for chunk_idx, doc in enumerate(docs):
            vectors = self.artifact.vectors_for_doc(doc)
            window_cacheable_ids = self.artifact.window_cacheable_ids_for_doc(doc)
            for cacheable_idx, cacheable in enumerate(
                getattr(doc, "cacheables", []) or []
            ):
                if not getattr(cacheable, "text", None):
                    continue
                cacheable_id = getattr(cacheable, "id", None)
                candidate = {
                    "chunk_idx": chunk_idx,
                    "cacheable_idx": cacheable_idx,
                    "cacheable_id": cacheable_id,
                    "vectors": (
                        vectors[cacheable_idx]
                        if cacheable_idx < len(vectors)
                        else torch.empty((0, 0))
                    ),
                    "window_cacheable_ids": (
                        window_cacheable_ids[cacheable_idx]
                        if cacheable_idx < len(window_cacheable_ids)
                        else [cacheable_id]
                    ),
                }
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _deduplicate_scored_candidates(scored_candidates):
        best_by_id = {}
        deduped = []
        for score, candidate in scored_candidates:
            cacheable_id = candidate.get("cacheable_id")
            if not cacheable_id:
                deduped.append((score, candidate))
                continue
            existing = best_by_id.get(cacheable_id)
            if existing is None or score > existing[0]:
                best_by_id[cacheable_id] = (score, candidate)
        deduped.extend(best_by_id.values())
        deduped.sort(key=lambda item: item[0], reverse=True)
        return deduped

    def _score_candidate(self, query_vector: torch.Tensor, candidate) -> float:
        return score_maxsim(query_vector, candidate["vectors"])

    @staticmethod
    def _score_coarse_chunk(
        query_vector: torch.Tensor, vectors: list[torch.Tensor]
    ) -> float:
        nonempty_vectors = [vector for vector in vectors if vector.numel() > 0]
        if not nonempty_vectors:
            return float("-inf")
        return score_maxsim(query_vector, torch.cat(nonempty_vectors, dim=0))

    def _rerank_chunk_indices(
        self,
        docs: List[RetrievableChunk],
        query_vector: torch.Tensor,
        profile: dict[str, float | int] | None = None,
    ) -> list[int]:
        scored_chunks = []
        for chunk_idx, doc in enumerate(docs):
            lookup_start = time.perf_counter()
            vectors = self.artifact.vectors_for_doc(doc)
            self._profile_add(
                profile, "artifact_lookup_time", time.perf_counter() - lookup_start
            )
            score_start = time.perf_counter()
            score = self._score_coarse_chunk(query_vector, vectors)
            self._profile_add(
                profile,
                "coarse_rerank_score_time",
                time.perf_counter() - score_start,
            )
            scored_chunks.append((score, chunk_idx))

        sort_start = time.perf_counter()
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        self._profile_add(
            profile, "coarse_rerank_sort_time", time.perf_counter() - sort_start
        )
        return [chunk_idx for _, chunk_idx in scored_chunks]

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            scored_candidates = [
                (self._score_candidate(query_vector, candidate), candidate)
                for candidate in self._iter_candidates(docs)
            ]
            scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
            selected_by_doc: dict[int, list[int]] = {}
            final_token_budget = self._resolve_final_token_budget(
                docs, getattr(self, "final_token_budget", None)
            )
            if final_token_budget is None:
                keep_count = global_top_count(len(scored_candidates), self.global_top_r)
                selected_candidates = scored_candidates[:keep_count]
            else:
                selected_candidates = []
                selected_ids = set()
                used_tokens = 0
                for score, candidate in scored_candidates:
                    cacheable_id = candidate.get("cacheable_id")
                    if cacheable_id in selected_ids:
                        continue
                    chunk_idx = candidate["chunk_idx"]
                    cacheable_idx = candidate["cacheable_idx"]
                    cacheables = getattr(docs[chunk_idx], "cacheables", []) or []
                    if not (0 <= cacheable_idx < len(cacheables)):
                        continue
                    token_len = self._cacheable_token_len(cacheables[cacheable_idx])
                    if token_len > final_token_budget:
                        continue
                    if selected_ids and used_tokens + token_len > final_token_budget:
                        continue
                    selected_ids.add(cacheable_id)
                    selected_candidates.append((score, candidate))
                    used_tokens += token_len
                    if used_tokens >= final_token_budget:
                        break

            for _, candidate in selected_candidates:
                selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                    candidate["cacheable_idx"]
                )

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class ColBERTRerankSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        keep = _configured_colbert_rerank_keep()
        super().__init__(
            initialize_global_top_r=False,
            initialize_token_budget=False,
        )
        self.keep = keep

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = {
            "query_encode_time": 0.0,
            "artifact_lookup_time": 0.0,
            "coarse_rerank_score_time": 0.0,
            "coarse_rerank_sort_time": 0.0,
            "build_output_time": 0.0,
            "query_count": len(batch_queries),
            "retrieved_doc_count": sum(len(docs) for docs in batch_top_k_docs),
            "selected_doc_count": 0,
        }
        start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - start

        summarized_batches = []
        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            ranked_indices = self._rerank_chunk_indices(docs, query_vector, profile)
            selected_indices = ranked_indices[: min(self.keep, len(ranked_indices))]

            build_start = time.perf_counter()
            summarized_batches.append([docs[idx].clone() for idx in selected_indices])
            profile["selected_doc_count"] += len(selected_indices)
            profile["build_output_time"] += time.perf_counter() - build_start

        self.last_profile = profile
        return summarized_batches


class BudgetColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__(
            initialize_global_top_r=False,
            initialize_token_budget=True,
        )
        if self.final_token_budget is None and self.retain_token_ratio is None:
            raise ValueError(
                "colbert_window_budget requires RETAIN_TOKEN_RATIO or COLBERT_FINAL_TOKEN_BUDGET"
            )

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        return super().compress_batch_top_k_docs(batch_top_k_docs, batch_queries)


class SlidingRegionColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self._sliding_region_spec_cache = {}
        self.region_token_budget = int(self.artifact.index["window_token_budget"])
        if self.region_token_budget <= self.encoder.doc_token_overhead:
            raise ValueError(
                "sliding region token budget must be larger than ColBERT document token overhead, "
                f"got {self.region_token_budget}"
            )
        retrieval_chunk_size = _resolve_configured_retrieval_chunk_size()
        if retrieval_chunk_size <= self.region_token_budget + 100:
            raise ValueError(
                "retrieval chunk size must leave a tokenizer-safety buffer above "
                "the ColBERT region budget: "
                f"retrieval_chunk_size={retrieval_chunk_size}, "
                f"region_token_budget={self.region_token_budget}, buffer=100"
            )

    def clear_inter_batch_cache(self) -> None:
        super().clear_inter_batch_cache()
        self._sliding_region_spec_cache.clear()

    @staticmethod
    def _build_region_document(
        doc: RetrievableChunk, selected_cacheables: list[CacheableChunk]
    ) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = [
            cacheable.clone()
            for _, cacheable in sorted(
                enumerate(selected_cacheables),
                key=lambda item: (
                    item[1].chunk_start is None,
                    item[1].chunk_start if item[1].chunk_start is not None else item[0],
                    item[1].chunk_end if item[1].chunk_end is not None else item[0],
                    item[0],
                ),
            )
        ]
        return cloned

    def _sliding_region_cache_key(
        self, doc: RetrievableChunk, cacheables: list[CacheableChunk]
    ):
        return (
            str(doc.id),
            self.region_token_budget,
            tuple(str(getattr(cacheable, "id", "")) for cacheable in cacheables),
        )

    def _cached_sliding_region_specs(
        self, doc: RetrievableChunk, cacheables: list[CacheableChunk]
    ):
        cache_key = self._sliding_region_cache_key(doc, cacheables)
        cached = self._sliding_region_spec_cache.get(cache_key)
        if cached is not None:
            return cached

        region_specs = []
        sidecar_specs = self.artifact.region_specs_for_doc(
            doc, self.region_token_budget
        )
        for center_idx, selected_indices in sidecar_specs:
            if not selected_indices:
                continue
            region_specs.append((center_idx, selected_indices))

        self._sliding_region_spec_cache[cache_key] = region_specs
        return region_specs

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
        lookup_start = time.perf_counter()
        vectors = self.artifact.vectors_for_doc(doc)
        self._profile_add(
            profile, "artifact_lookup_time", time.perf_counter() - lookup_start
        )
        spec_start = time.perf_counter()
        region_specs = self._cached_sliding_region_specs(doc, cacheables)
        self._profile_add(profile, "region_spec_time", time.perf_counter() - spec_start)
        regions = []
        object_start = time.perf_counter()
        for center_idx, selected_indices in region_specs:
            regions.append(
                {
                    "chunk_idx": chunk_idx,
                    "center_idx": center_idx,
                    "cacheable": None,
                    "region_id": f"{doc.id}::sliding_region_{center_idx}",
                    "parent_doc_id": doc.id,
                    "source_doc": doc,
                    "selected_indices": selected_indices,
                    "source_cacheables": cacheables,
                    "source_vectors": vectors,
                }
            )
        self._profile_add(
            profile, "region_object_time", time.perf_counter() - object_start
        )
        self._profile_add(profile, "region_count", len(regions))
        return regions

    @staticmethod
    def _sentence_score_cache_key(region, idx: int):
        cacheables = region["source_cacheables"]
        raw_cacheable_id = getattr(cacheables[idx], "id", None)
        source_vectors = region.get("source_vectors")
        source_key = id(source_vectors) if source_vectors is not None else "no_source"
        if raw_cacheable_id:
            return (source_key, idx, str(raw_cacheable_id))
        return (source_key, region["chunk_idx"], idx)

    def _score_sliding_regions_vectorized(
        self,
        query_vector: torch.Tensor,
        regions,
        profile: dict[str, float | int] | None = None,
    ) -> list[float]:
        if not regions:
            return []

        collect_start = time.perf_counter()
        sentence_index_by_key: dict[tuple, int] = {}
        sentence_vectors = []
        region_sentence_indices = []
        fallback_regions = []

        for region_idx, region in enumerate(regions):
            if region.get("source_vectors") is None:
                fallback_regions.append(region_idx)
                region_sentence_indices.append([])
                continue
            indices = []
            source_vectors = region["source_vectors"]
            cacheables = region["source_cacheables"]
            for idx in region["selected_indices"]:
                if not (0 <= idx < len(cacheables)):
                    continue
                cache_key = self._sentence_score_cache_key(region, idx)
                sentence_idx = sentence_index_by_key.get(cache_key)
                if sentence_idx is None:
                    vectors = (
                        source_vectors[idx]
                        if idx < len(source_vectors)
                        else torch.empty((0, 0))
                    )
                    sentence_idx = len(sentence_vectors)
                    sentence_index_by_key[cache_key] = sentence_idx
                    sentence_vectors.append(vectors)
                indices.append(sentence_idx)
            region_sentence_indices.append(indices)

        if not sentence_vectors:
            scores = [float("-inf")] * len(regions)
            for region_idx in fallback_regions:
                region = regions[region_idx]
                scores[region_idx] = score_maxsim(
                    query_vector, region.get("vectors", torch.empty((0, 0)))
                )
            return scores

        self._profile_add(profile, "unique_sentence_count", len(sentence_vectors))
        sentence_lengths = torch.tensor(
            [int(vectors.shape[0]) for vectors in sentence_vectors],
            dtype=torch.long,
        )
        nonempty_items = [
            (idx, vectors)
            for idx, vectors in enumerate(sentence_vectors)
            if vectors.numel() > 0
        ]
        if nonempty_items:
            sentence_ids = torch.repeat_interleave(
                torch.tensor(
                    [idx for idx, _ in nonempty_items],
                    dtype=torch.long,
                    device=query_vector.device,
                ),
                torch.tensor(
                    [int(vectors.shape[0]) for _, vectors in nonempty_items],
                    dtype=torch.long,
                    device=query_vector.device,
                ),
            )
            all_vectors = torch.cat(
                [vectors for _, vectors in nonempty_items], dim=0
            ).to(query_vector.device)
        else:
            sentence_ids = torch.empty(
                (0,), dtype=torch.long, device=query_vector.device
            )
            dim = int(query_vector.shape[1]) if query_vector.dim() == 2 else 0
            all_vectors = torch.empty(
                (0, dim), dtype=torch.float32, device=query_vector.device
            )
        self._profile_add(
            profile, "sentence_token_count", int(sentence_lengths.sum().item())
        )
        self._profile_add(
            profile, "sentence_maxsim_time", time.perf_counter() - collect_start
        )

        maxsim_start = time.perf_counter()
        query_float = query_vector.to(torch.float32)
        sentence_scores = torch.full(
            (len(sentence_vectors), query_float.shape[0]),
            float("-inf"),
            dtype=torch.float32,
            device=query_float.device,
        )
        if all_vectors.numel() > 0:
            sims = torch.matmul(query_float, all_vectors.to(torch.float32).T)
            sentence_scores_t = sentence_scores.T.contiguous()
            index = sentence_ids.unsqueeze(0).expand(query_float.shape[0], -1)
            sentence_scores_t.scatter_reduce_(
                1, index, sims, reduce="amax", include_self=True
            )
            sentence_scores = sentence_scores_t.T.contiguous()
        self._profile_add(
            profile, "sentence_maxsim_time", time.perf_counter() - maxsim_start
        )

        region_start = time.perf_counter()
        max_region_sentences = max(
            (len(indices) for indices in region_sentence_indices), default=0
        )
        if max_region_sentences == 0:
            scores = [float("-inf")] * len(regions)
        else:
            region_index_tensor = torch.full(
                (len(regions), max_region_sentences),
                -1,
                dtype=torch.long,
                device=sentence_scores.device,
            )
            for region_idx, indices in enumerate(region_sentence_indices):
                if indices:
                    region_index_tensor[region_idx, : len(indices)] = torch.tensor(
                        indices, dtype=torch.long, device=sentence_scores.device
                    )
            valid = region_index_tensor >= 0
            gathered = sentence_scores[region_index_tensor.clamp_min(0)]
            gathered = gathered.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            region_scores = gathered.max(dim=1).values.sum(dim=1)
            scores = [float(value) for value in region_scores.detach().cpu().tolist()]

        for region_idx in fallback_regions:
            region = regions[region_idx]
            scores[region_idx] = score_maxsim(
                query_vector, region.get("vectors", torch.empty((0, 0)))
            )
        self._profile_add(
            profile, "region_score_time", time.perf_counter() - region_start
        )
        return scores

    def _region_sentence_indices_to_keep(
        self, query_vector: torch.Tensor | None, region
    ) -> set[int]:
        return set(region["selected_indices"])

    def _make_region_run_cacheables(self, region, selected_indices):
        runs = []
        current_run = []
        previous_idx = None
        for idx in selected_indices:
            if previous_idx is not None and idx != previous_idx + 1:
                if current_run:
                    runs.append(current_run)
                current_run = []
            current_run.append(idx)
            previous_idx = idx
        if current_run:
            runs.append(current_run)

        cacheables = []
        source_cacheables = region["source_cacheables"]
        for run in runs:
            run_cacheables = [source_cacheables[idx] for idx in run]
            first = run_cacheables[0]
            last = run_cacheables[-1]
            cacheables.append(
                CacheableChunk(
                    id=f"{region['region_id']}::dedup_{run[0]}_{run[-1]}",
                    text=" ".join(cacheable.text for cacheable in run_cacheables),
                    parent_doc_id=region["parent_doc_id"],
                    chunk_size=self.region_token_budget,
                    sentence_ids=[cacheable.id for cacheable in run_cacheables],
                    sentence_texts=[cacheable.text for cacheable in run_cacheables],
                    chunk_start=first.chunk_start,
                    chunk_end=last.chunk_end,
                )
            )
        return cacheables

    def _select_sliding_regions(
        self,
        scored_regions,
        query_vector: torch.Tensor | None = None,
        final_token_budget: int | None = None,
    ):
        if final_token_budget is None:
            keep_count = global_top_count(len(scored_regions), self.global_top_r)
            selected_cacheables = []
            selected_sentence_ids = set()
            for _, region in scored_regions[:keep_count]:
                novel_indices = []
                for idx in region["selected_indices"]:
                    source = region["source_cacheables"][idx]
                    if source.id in selected_sentence_ids:
                        continue
                    novel_indices.append(idx)
                    selected_sentence_ids.add(source.id)
                selected_cacheables.extend(
                    self._make_region_run_cacheables(region, novel_indices)
                )
            return selected_cacheables

        self._cacheable_token_lens(
            self._unique_cacheables(
                [
                    source
                    for _, region in scored_regions
                    for source in region["source_cacheables"]
                ]
            )
        )
        selected_cacheables = []
        selected_sentence_ids = set()
        used_tokens = 0
        for _, region in scored_regions:
            novel_indices = []
            novel_token_count = 0
            allowed_indices = self._region_sentence_indices_to_keep(
                query_vector, region
            )
            for idx in region["selected_indices"]:
                if idx not in allowed_indices:
                    continue
                source = region["source_cacheables"][idx]
                if source.id in selected_sentence_ids:
                    continue
                token_len = self._cacheable_token_len(source)
                novel_indices.append(idx)
                novel_token_count += token_len

            if not novel_indices:
                continue

            for idx in novel_indices:
                selected_sentence_ids.add(region["source_cacheables"][idx].id)
            used_tokens += novel_token_count

            selected_cacheables.extend(
                (region["chunk_idx"], cacheable)
                for cacheable in self._make_region_run_cacheables(region, novel_indices)
            )

            if used_tokens >= final_token_budget:
                break

        return selected_cacheables

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = self._empty_profile()
        profile["query_count"] = len(batch_queries)
        profile["retrieved_doc_count"] = sum(len(docs) for docs in batch_top_k_docs)
        query_encode_start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - query_encode_start
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            budget_start = time.perf_counter()
            final_token_budget = self._resolve_final_token_budget(
                docs, getattr(self, "final_token_budget", None)
            )
            self._profile_add(
                profile, "budget_time", time.perf_counter() - budget_start
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx, profile))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            region_scores = self._score_sliding_regions_vectorized(
                query_vector, regions, profile
            )
            scored_regions = list(zip(region_scores, regions))
            sort_start = time.perf_counter()
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            self._profile_add(profile, "sort_time", time.perf_counter() - sort_start)
            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            select_start = time.perf_counter()
            if final_token_budget is None:
                for cacheable in self._select_sliding_regions(
                    scored_regions,
                    query_vector,
                    final_token_budget=final_token_budget,
                ):
                    doc_id = str(cacheable.parent_doc_id)
                    for chunk_idx, doc in enumerate(docs):
                        if str(doc.id) == doc_id:
                            selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
                            break
            else:
                for chunk_idx, cacheable in self._select_sliding_regions(
                    scored_regions,
                    query_vector,
                    final_token_budget=final_token_budget,
                ):
                    selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
            self._profile_add(
                profile, "select_time", time.perf_counter() - select_start
            )

            output_start = time.perf_counter()
            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            self._profile_add(
                profile, "build_output_time", time.perf_counter() - output_start
            )
            summarized_batches.append(summarized_docs)

        self.last_profile = profile
        return summarized_batches


class ColBERTRerankAndRegionSummarizer(SlidingRegionColBERTWindowSummarizer):
    def __init__(self):
        self.rerank_keep = _configured_colbert_rerank_keep()
        super().__init__()

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = self._empty_profile()
        profile["query_count"] = len(batch_queries)
        profile["retrieved_doc_count"] = sum(len(docs) for docs in batch_top_k_docs)
        query_encode_start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - query_encode_start
        summarized_batches = []
        rerank_kept_chunk_count = 0

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            ranked_indices = self._rerank_chunk_indices(docs, query_vector, profile)
            allowed_indices = set(ranked_indices[: self.rerank_keep])
            rerank_kept_chunk_count += len(allowed_indices)
            gated_docs = [doc for idx, doc in enumerate(docs) if idx in allowed_indices]

            budget_start = time.perf_counter()
            final_token_budget = self._resolve_final_token_budget(
                gated_docs, getattr(self, "final_token_budget", None)
            )
            self._profile_add(
                profile, "budget_time", time.perf_counter() - budget_start
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                if chunk_idx not in allowed_indices:
                    continue
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx, profile))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            region_scores = self._score_sliding_regions_vectorized(
                query_vector, regions, profile
            )
            scored_regions = list(zip(region_scores, regions))
            sort_start = time.perf_counter()
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            self._profile_add(profile, "sort_time", time.perf_counter() - sort_start)

            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            select_start = time.perf_counter()
            if final_token_budget is None:
                for cacheable in self._select_sliding_regions(
                    scored_regions,
                    query_vector,
                    final_token_budget=final_token_budget,
                ):
                    doc_id = str(cacheable.parent_doc_id)
                    for chunk_idx, doc in enumerate(docs):
                        if str(doc.id) == doc_id:
                            selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
                            break
            else:
                for chunk_idx, cacheable in self._select_sliding_regions(
                    scored_regions,
                    query_vector,
                    final_token_budget=final_token_budget,
                ):
                    selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
            self._profile_add(
                profile, "select_time", time.perf_counter() - select_start
            )

            output_start = time.perf_counter()
            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            self._profile_add(
                profile, "build_output_time", time.perf_counter() - output_start
            )
            summarized_batches.append(summarized_docs)

        profile["rerank_kept_chunk_count"] = rerank_kept_chunk_count
        self.last_profile = profile
        return summarized_batches
