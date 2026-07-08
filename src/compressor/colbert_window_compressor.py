import math
import os
from typing import List

import torch

from chunk import CacheableChunk, RetrievableChunk
from compressor.base import Compressor
from compressor.comparison_compressor import GlobalComparisonSummarizer
from materialize.colbert_window import (
    ColBERTWindowArtifact,
    ColBERTWindowEncoder,
    WindowSpec,
    default_colbert_repo_path,
    global_top_count,
    parse_bool,
    score_maxsim,
)


class ColBERTWindowSummarizer(Compressor):
    def __init__(self):
        super().__init__()
        self.global_top_r = GlobalComparisonSummarizer._parse_global_top_r(
            os.getenv("GLOBAL_TOP_R", "0.1")
        )
        retain_token_ratio = os.getenv("RETAIN_TOKEN_RATIO")
        self.retain_token_ratio = (
            self._parse_retain_token_ratio(retain_token_ratio)
            if retain_token_ratio is not None
            else None
        )
        self.length_penalty = (
            os.getenv("COLBERT_LENGTH_PENALTY", "none").strip().lower()
        )
        if self.length_penalty not in {"none", "sqrt", "log", "sub_log"}:
            raise ValueError(
                "COLBERT_LENGTH_PENALTY must be one of {'none', 'sqrt', 'log', 'sub_log'}, "
                f"got {self.length_penalty!r}"
            )
        self.length_penalty_alpha = float(
            os.getenv("COLBERT_LENGTH_PENALTY_ALPHA", "0.1")
        )
        self._token_len_cache: dict[str, int] = {}
        artifact_dir = os.getenv("COLBERT_WINDOW_DIR")
        if not artifact_dir:
            dataset_path = os.getenv("DATASET_PATH")
            data_subdir = os.getenv("DATA_SUBDIR", "sent")
            if not dataset_path:
                raise ValueError(
                    "COLBERT_WINDOW_DIR or DATASET_PATH must be set for colbert_window compression"
                )
            artifact_dir = os.path.join(dataset_path, data_subdir, "colbert_window")

        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        device = os.getenv("COLBERT_DEVICE") or "cpu"
        batch_size = int(os.getenv("COLBERT_BATCH_SIZE", "32"))
        query_maxlen_env = os.getenv("COLBERT_QUERY_MAXLEN")
        query_maxlen = int(query_maxlen_env) if query_maxlen_env else None
        attend_to_mask_tokens_env = os.getenv("COLBERT_ATTEND_TO_MASK_TOKENS")
        attend_to_mask_tokens = (
            parse_bool(attend_to_mask_tokens_env)
            if attend_to_mask_tokens_env is not None
            else None
        )
        if self.length_penalty != "none":
            print(
                "ColBERT length penalty enabled: "
                f"{self.length_penalty}, alpha={self.length_penalty_alpha}"
            )
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()
        disable_cpu_extension = parse_bool(
            os.getenv("COLBERT_DISABLE_CPU_EXTENSION", "True")
        )

        print(f"ColBERT window compression enabled. Loading artifact: {artifact_dir}")
        self.artifact = ColBERTWindowArtifact(artifact_dir)
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
            device=device,
            batch_size=batch_size,
            max_length=int(self.artifact.index.get("official_doc_maxlen", 0)),
            query_maxlen=query_maxlen,
            attend_to_mask_tokens=attend_to_mask_tokens,
            disable_cpu_extension=disable_cpu_extension,
            verify_tensorization=False,
        )

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
        total = 0
        seen_ids = set()
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
                total += self._cacheable_token_len(cacheable)
        return total

    def _resolve_final_token_budget(
        self, docs: List[RetrievableChunk], absolute_budget: int | None
    ) -> int | None:
        if self.retain_token_ratio is None:
            return absolute_budget
        retrieved_tokens = self._retrieved_context_token_count(docs)
        if retrieved_tokens <= 0:
            return 0
        return max(1, math.ceil(retrieved_tokens * self.retain_token_ratio))

    @staticmethod
    def _build_unselected_document(doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned

    def _cacheable_token_len(self, cacheable) -> int:
        cacheable_id = getattr(cacheable, "id", None)
        if cacheable_id:
            cached = self._token_len_cache.get(cacheable_id)
            if cached is not None:
                return cached
        token_len = self.encoder.token_count(cacheable.text)
        if cacheable_id:
            self._token_len_cache[cacheable_id] = token_len
        return token_len

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

    @staticmethod
    def _candidate_token_len(candidate) -> int:
        vectors = candidate.get("vectors")
        if isinstance(vectors, torch.Tensor) and vectors.dim() >= 1:
            return max(1, int(vectors.shape[0]))
        return 1

    def _apply_length_penalty(self, score: float, token_len: int) -> float:
        if self.length_penalty == "none":
            return float(score)
        token_len = max(1, int(token_len))
        if self.length_penalty == "sqrt":
            return float(score) / math.sqrt(token_len)
        if self.length_penalty == "log":
            return float(score) / math.log1p(token_len)
        if self.length_penalty == "sub_log":
            return float(score) - self.length_penalty_alpha * math.log1p(token_len)
        raise AssertionError(f"unhandled length penalty: {self.length_penalty}")

    def _score_candidate(self, query_vector: torch.Tensor, candidate) -> float:
        score = score_maxsim(query_vector, candidate["vectors"])
        return self._apply_length_penalty(score, self._candidate_token_len(candidate))

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
            keep_count = global_top_count(len(scored_candidates), self.global_top_r)
            if keep_count == 0:
                summarized_batches.append(summarized_docs)
                continue

            selected_by_doc: dict[int, list[int]] = {}
            for _, candidate in scored_candidates[:keep_count]:
                selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                    candidate["cacheable_idx"]
                )

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class BudgetColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        final_budget = os.getenv("COLBERT_FINAL_TOKEN_BUDGET")
        if not final_budget and self.retain_token_ratio is None:
            raise ValueError(
                "colbert_window_budget requires RETAIN_TOKEN_RATIO or COLBERT_FINAL_TOKEN_BUDGET"
            )
        self.final_token_budget = int(final_budget) if final_budget else None
        if self.final_token_budget is not None and self.final_token_budget <= 0:
            raise ValueError(
                f"COLBERT_FINAL_TOKEN_BUDGET must be positive, got {self.final_token_budget}"
            )

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(
                docs, self.final_token_budget
            )
            if final_token_budget is None or final_token_budget <= 0:
                summarized_batches.append(summarized_docs)
                continue
            scored_candidates = [
                (self._score_candidate(query_vector, candidate), candidate)
                for candidate in self._iter_candidates(docs)
            ]
            scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
            selected_by_doc: dict[int, list[int]] = {}
            selected_ids = set()
            used_tokens = 0
            for _, candidate in scored_candidates:
                cacheable_id = candidate.get("cacheable_id")
                if cacheable_id in selected_ids:
                    continue
                chunk_idx = candidate["chunk_idx"]
                cacheable_idx = candidate["cacheable_idx"]
                cacheables = getattr(docs[chunk_idx], "cacheables", []) or []
                if not (0 <= cacheable_idx < len(cacheables)):
                    continue
                token_len = self._cacheable_token_len(cacheables[cacheable_idx])
                if selected_ids and used_tokens + token_len > final_token_budget:
                    continue
                selected_ids.add(cacheable_id)
                selected_by_doc.setdefault(chunk_idx, []).append(cacheable_idx)
                used_tokens += token_len
                if used_tokens >= final_token_budget:
                    break

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class FixedRegionColBERTSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.region_token_budget = 220
        final_budget = os.getenv("COLBERT_FINAL_TOKEN_BUDGET")
        self.final_token_budget = int(final_budget) if final_budget else None
        if self.region_token_budget <= self.encoder.doc_token_overhead:
            raise ValueError(
                "fixed region token budget must be larger than ColBERT document token overhead, "
                f"got {self.region_token_budget}"
            )
        if self.final_token_budget is not None and self.final_token_budget <= 0:
            raise ValueError(
                f"COLBERT_FINAL_TOKEN_BUDGET must be positive, got {self.final_token_budget}"
            )

    def _fixed_regions_for_doc(self, doc: RetrievableChunk):
        encoded = self.encoder.doc_tokenizer(
            doc.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        body_budget = max(1, self.region_token_budget - self.encoder.doc_token_overhead)
        regions = []
        for region_idx, start in enumerate(range(0, len(token_ids), body_budget)):
            end = min(start + body_budget, len(token_ids))
            region_offsets = [
                offset for offset in offsets[start:end] if offset[1] > offset[0]
            ]
            if not region_offsets:
                continue
            char_start = int(region_offsets[0][0])
            char_end = int(region_offsets[-1][1])
            text = doc.text[char_start:char_end].strip()
            if not text:
                continue
            cacheable = CacheableChunk(
                id=f"{doc.id}::fixed_region_{region_idx}",
                text=text,
                parent_doc_id=doc.id,
                chunk_size=self.region_token_budget,
                chunk_start=char_start,
                chunk_end=char_end,
            )
            spec = WindowSpec(
                text=text,
                center_start=0,
                center_end=len(text),
                selected_indices=[region_idx],
                addition_order=[region_idx],
                truncated_center=False,
            )
            regions.append((region_idx, cacheable, spec))
        return regions

    @staticmethod
    def _build_region_document(
        doc: RetrievableChunk, selected_cacheables: list[CacheableChunk]
    ) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = [cacheable.clone() for cacheable in selected_cacheables]
        return cloned

    def _select_fixed_regions(self, scored_regions, final_token_budget: int | None):
        if final_token_budget is None:
            keep_count = global_top_count(len(scored_regions), self.global_top_r)
            return [ref for _, ref in scored_regions[:keep_count]]

        selected = []
        used_tokens = 0
        for _, ref in scored_regions:
            _, _, cacheable = ref
            token_len = self._cacheable_token_len(cacheable)
            if selected and used_tokens + token_len > final_token_budget:
                continue
            selected.append(ref)
            used_tokens += token_len
            if used_tokens >= final_token_budget:
                break
        return selected

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(
                docs, self.final_token_budget
            )
            specs = []
            region_refs = []
            for chunk_idx, doc in enumerate(docs):
                for region_idx, cacheable, spec in self._fixed_regions_for_doc(doc):
                    specs.append(spec)
                    region_refs.append((chunk_idx, region_idx, cacheable))
            if not specs:
                summarized_batches.append(summarized_docs)
                continue

            vectors = self.encoder.encode_windows(specs)
            scored_regions = [
                (score_maxsim(query_vector, region_vectors), ref)
                for region_vectors, ref in zip(vectors, region_refs)
            ]
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            for chunk_idx, _, cacheable in self._select_fixed_regions(
                scored_regions, final_token_budget
            ):
                selected_by_doc.setdefault(chunk_idx, []).append(cacheable)

            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class SlidingRegionColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        window_budget = os.getenv("COLBERT_SLIDING_WINDOW_TOKEN_BUDGET")
        artifact_window_budget = int(
            self.artifact.index.get("window_token_budget") or self.encoder.max_length
        )
        self.region_token_budget = (
            int(window_budget) if window_budget else artifact_window_budget
        )
        final_budget = os.getenv("COLBERT_FINAL_TOKEN_BUDGET")
        self.final_token_budget = int(final_budget) if final_budget else None
        if self.region_token_budget <= self.encoder.doc_token_overhead:
            raise ValueError(
                "sliding region token budget must be larger than ColBERT document token overhead, "
                f"got {self.region_token_budget}"
            )
        if self.final_token_budget is not None and self.final_token_budget <= 0:
            raise ValueError(
                f"COLBERT_FINAL_TOKEN_BUDGET must be positive, got {self.final_token_budget}"
            )

    @staticmethod
    def _concat_candidate_vectors(candidates) -> torch.Tensor:
        vectors = [
            candidate["vectors"]
            for candidate in candidates
            if candidate["vectors"].numel() > 0
        ]
        if vectors:
            return torch.cat(vectors, dim=0)
        dim = 0
        for candidate in candidates:
            if candidate["vectors"].dim() == 2:
                dim = candidate["vectors"].shape[1]
                break
        return torch.empty((0, dim), dtype=torch.float16)

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
        vectors = self.artifact.vectors_for_doc(doc)
        specs = self.encoder.build_centered_windows(
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
                id=f"{doc.id}::sliding_region_{center_idx}",
                text=spec.text,
                parent_doc_id=doc.id,
                chunk_size=self.region_token_budget,
                sentence_ids=sentence_ids,
                sentence_texts=[cacheables[idx].text for idx in selected_indices],
            )
            regions.append(
                {
                    "chunk_idx": chunk_idx,
                    "center_idx": center_idx,
                    "cacheable": cacheable,
                    "selected_indices": selected_indices,
                    "source_cacheables": cacheables,
                    "source_vectors": vectors,
                }
            )
        return regions

    @staticmethod
    def _sentence_query_max_scores(
        query_vector: torch.Tensor, sentence_vectors: torch.Tensor
    ) -> torch.Tensor:
        if query_vector.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=query_vector.device)
        if sentence_vectors.numel() == 0:
            return torch.full(
                (query_vector.shape[0],),
                float("-inf"),
                dtype=torch.float32,
                device=query_vector.device,
            )
        sims = torch.matmul(
            query_vector.to(torch.float32), sentence_vectors.to(torch.float32).T
        )
        return sims.max(dim=1).values

    def _score_sliding_region(
        self,
        query_vector: torch.Tensor,
        region,
        sentence_score_cache: dict[str, torch.Tensor],
    ) -> float:
        source_vectors = region.get("source_vectors")
        if source_vectors is None:
            return score_maxsim(
                query_vector, region.get("vectors", torch.empty((0, 0)))
            )

        per_sentence_scores = []
        cacheables = region["source_cacheables"]
        for idx in region["selected_indices"]:
            if not (0 <= idx < len(cacheables)):
                continue
            raw_cacheable_id = getattr(cacheables[idx], "id", None)
            cacheable_id = f"{region['chunk_idx']}::{raw_cacheable_id or idx}"
            scores = sentence_score_cache.get(cacheable_id)
            if scores is None:
                vectors = (
                    source_vectors[idx]
                    if idx < len(source_vectors)
                    else torch.empty((0, 0))
                )
                scores = self._sentence_query_max_scores(query_vector, vectors)
                sentence_score_cache[cacheable_id] = scores
            per_sentence_scores.append(scores)

        if not per_sentence_scores:
            return float("-inf")
        stacked = torch.stack(per_sentence_scores, dim=0)
        return float(stacked.max(dim=0).values.sum().item())

    def _region_sentence_indices_to_keep(
        self, query_vector: torch.Tensor | None, region
    ) -> set[int]:
        return set(region["selected_indices"])

    def _select_sliding_regions(
        self,
        scored_regions,
        query_vector: torch.Tensor | None = None,
        final_token_budget: int | None = None,
    ):
        if final_token_budget is None:
            keep_count = global_top_count(len(scored_regions), self.global_top_r)
            return [region["cacheable"] for _, region in scored_regions[:keep_count]]

        selected_cacheables = []
        selected_sentence_ids = set()
        used_tokens = 0
        for _, region in scored_regions:
            novel_sentence_cacheables = []
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
                if (
                    novel_sentence_cacheables
                    and used_tokens + token_len > final_token_budget
                ):
                    continue
                if (
                    not novel_sentence_cacheables
                    and selected_cacheables
                    and used_tokens + token_len > final_token_budget
                ):
                    continue
                novel_sentence_cacheables.append(source)
                selected_sentence_ids.add(source.id)
                used_tokens += token_len
                if used_tokens >= final_token_budget:
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
                selected_cacheables.append((region["chunk_idx"], region_cacheable))

            if used_tokens >= final_token_budget:
                break

        return selected_cacheables

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(
                docs, self.final_token_budget
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            sentence_score_cache = {}
            scored_regions = [
                (
                    self._score_sliding_region(
                        query_vector, region, sentence_score_cache
                    ),
                    region,
                )
                for region in regions
            ]
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            selected_by_doc: dict[int, list[CacheableChunk]] = {}
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

            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class PrunedSlidingRegionColBERTWindowSummarizer(SlidingRegionColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.sentence_top_r = GlobalComparisonSummarizer._parse_global_top_r(
            os.getenv("COLBERT_REGION_PRUNE_SENTENCE_TOP_R", "0.5")
        )
        if self.sentence_top_r <= 0:
            raise ValueError(
                "COLBERT_REGION_PRUNE_SENTENCE_TOP_R must be positive, "
                f"got {self.sentence_top_r}"
            )
        print(
            "ColBERT sliding-region sentence pruning enabled: "
            f"sentence_top_r={self.sentence_top_r}"
        )

    def _region_sentence_indices_to_keep(
        self, query_vector: torch.Tensor | None, region
    ) -> set[int]:
        selected_indices = list(region["selected_indices"])
        if query_vector is None or not selected_indices:
            return set(selected_indices)

        source_vectors = region.get("source_vectors") or []
        scored_indices = []
        for idx in selected_indices:
            vectors = (
                source_vectors[idx]
                if idx < len(source_vectors)
                else torch.empty((0, 0))
            )
            scored_indices.append((score_maxsim(query_vector, vectors), idx))

        scored_indices.sort(key=lambda item: item[0], reverse=True)
        keep_count = global_top_count(len(scored_indices), self.sentence_top_r)
        return {idx for _, idx in scored_indices[:keep_count]}


class SupportPrunedSlidingRegionColBERTWindowSummarizer(
    SlidingRegionColBERTWindowSummarizer
):
    def __init__(self):
        super().__init__()
        support_top_r = os.getenv("COLBERT_SUPPORT_WINDOW_TOP_R")
        self.support_window_top_r = (
            GlobalComparisonSummarizer._parse_global_top_r(support_top_r)
            if support_top_r
            else self.global_top_r
        )
        if self.support_window_top_r <= 0:
            raise ValueError(
                "COLBERT_SUPPORT_WINDOW_TOP_R must be positive, "
                f"got {self.support_window_top_r}"
            )
        print(
            "ColBERT sliding-region support pruning enabled: "
            f"support_window_top_r={self.support_window_top_r}"
        )

    @staticmethod
    def _sentence_key(region, idx: int) -> str:
        return str(region["source_cacheables"][idx].id)

    def _rank_sentences_by_window_support(self, scored_regions):
        total_counts: dict[str, int] = {}
        sentence_refs = {}
        for region_score, region in scored_regions:
            for idx in region["selected_indices"]:
                key = self._sentence_key(region, idx)
                total_counts[key] = total_counts.get(key, 0) + 1
                sentence_refs.setdefault(
                    key, (region["chunk_idx"], region["source_cacheables"][idx])
                )

        support_region_count = global_top_count(
            len(scored_regions), self.support_window_top_r
        )
        selected_counts: dict[str, int] = {}
        best_region_scores: dict[str, float] = {}
        for region_score, region in scored_regions[:support_region_count]:
            for idx in region["selected_indices"]:
                key = self._sentence_key(region, idx)
                selected_counts[key] = selected_counts.get(key, 0) + 1
                best_region_scores[key] = max(
                    best_region_scores.get(key, float("-inf")), float(region_score)
                )

        ranked = []
        for key, selected_count in selected_counts.items():
            total_count = max(1, total_counts.get(key, 1))
            support_ratio = float(selected_count) / float(total_count)
            chunk_idx, cacheable = sentence_refs[key]
            ranked.append(
                (
                    support_ratio,
                    selected_count,
                    best_region_scores.get(key, float("-inf")),
                    chunk_idx,
                    cacheable,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return ranked

    def _select_supported_sentences(
        self, ranked_sentences, final_token_budget: int | None
    ):
        selected = []
        used_tokens = 0
        selected_sentence_ids = set()
        keep_count = (
            None
            if final_token_budget is not None
            else global_top_count(
                len(ranked_sentences),
                self.global_top_r,
            )
        )
        for _, _, _, chunk_idx, cacheable in ranked_sentences:
            if cacheable.id in selected_sentence_ids:
                continue
            if keep_count is not None and len(selected) >= keep_count:
                break
            token_len = self._cacheable_token_len(cacheable)
            if (
                final_token_budget is not None
                and selected
                and used_tokens + token_len > final_token_budget
            ):
                continue
            selected.append((chunk_idx, cacheable))
            selected_sentence_ids.add(cacheable.id)
            used_tokens += token_len
            if final_token_budget is not None and used_tokens >= final_token_budget:
                break
        return selected

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(
                docs, self.final_token_budget
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            sentence_score_cache = {}
            scored_regions = [
                (
                    self._score_sliding_region(
                        query_vector, region, sentence_score_cache
                    ),
                    region,
                )
                for region in regions
            ]
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            ranked_sentences = self._rank_sentences_by_window_support(scored_regions)
            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            for chunk_idx, cacheable in self._select_supported_sentences(
                ranked_sentences, final_token_budget
            ):
                selected_by_doc.setdefault(chunk_idx, []).append(cacheable)

            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class SupportCleanupSlidingRegionColBERTWindowSummarizer(
    SlidingRegionColBERTWindowSummarizer
):
    def __init__(self):
        super().__init__()
        self.support_ratio_threshold = float(
            os.getenv("COLBERT_SUPPORT_CLEANUP_MAX_RATIO", "0.2")
        )
        self.support_selected_count = int(
            os.getenv("COLBERT_SUPPORT_CLEANUP_SELECTED_COUNT", "1")
        )
        if self.support_ratio_threshold < 0:
            raise ValueError(
                "COLBERT_SUPPORT_CLEANUP_MAX_RATIO must be non-negative, "
                f"got {self.support_ratio_threshold}"
            )
        if self.support_selected_count <= 0:
            raise ValueError(
                "COLBERT_SUPPORT_CLEANUP_SELECTED_COUNT must be positive, "
                f"got {self.support_selected_count}"
            )
        print(
            "ColBERT sliding-region support cleanup enabled: "
            f"included_selected=={self.support_selected_count}, "
            f"support_ratio<={self.support_ratio_threshold}"
        )

    @staticmethod
    def _sentence_key(region, idx: int) -> str:
        return str(region["source_cacheables"][idx].id)

    def _cleanup_sentence_ids(self, scored_regions) -> set[str]:
        keep_count = global_top_count(len(scored_regions), self.global_top_r)
        selected_regions = scored_regions[:keep_count]
        total_counts: dict[str, int] = {}
        selected_counts: dict[str, int] = {}

        for _, region in scored_regions:
            for idx in region["selected_indices"]:
                key = self._sentence_key(region, idx)
                total_counts[key] = total_counts.get(key, 0) + 1

        for _, region in selected_regions:
            for idx in region["selected_indices"]:
                key = self._sentence_key(region, idx)
                selected_counts[key] = selected_counts.get(key, 0) + 1

        cleanup_ids = set()
        for key, selected_count in selected_counts.items():
            total_count = max(1, total_counts.get(key, 1))
            support_ratio = float(selected_count) / float(total_count)
            if (
                selected_count == self.support_selected_count
                and support_ratio <= self.support_ratio_threshold
            ):
                cleanup_ids.add(key)
        return cleanup_ids

    def _select_sliding_regions_with_cleanup(
        self,
        scored_regions,
        cleanup_ids: set[str],
        final_token_budget: int | None,
    ):
        if final_token_budget is None:
            keep_count = global_top_count(len(scored_regions), self.global_top_r)
            selected = []
            for _, region in scored_regions[:keep_count]:
                kept_indices = [
                    idx
                    for idx in region["selected_indices"]
                    if self._sentence_key(region, idx) not in cleanup_ids
                ]
                if not kept_indices:
                    continue
                cacheable = CacheableChunk(
                    id=f"{region['cacheable'].id}::support_cleanup",
                    text=" ".join(
                        region["source_cacheables"][idx].text for idx in kept_indices
                    ),
                    parent_doc_id=region["cacheable"].parent_doc_id,
                    chunk_size=self.region_token_budget,
                    sentence_ids=[
                        region["source_cacheables"][idx].id for idx in kept_indices
                    ],
                    sentence_texts=[
                        region["source_cacheables"][idx].text for idx in kept_indices
                    ],
                )
                selected.append(cacheable)
            return selected

        selected_cacheables = []
        selected_sentence_ids = set()
        used_tokens = 0
        for _, region in scored_regions:
            novel_sentence_cacheables = []
            for idx in region["selected_indices"]:
                source = region["source_cacheables"][idx]
                if source.id in selected_sentence_ids:
                    continue
                token_len = self._cacheable_token_len(source)
                if (
                    novel_sentence_cacheables
                    and used_tokens + token_len > final_token_budget
                ):
                    continue
                if (
                    not novel_sentence_cacheables
                    and selected_cacheables
                    and used_tokens + token_len > final_token_budget
                ):
                    continue
                selected_sentence_ids.add(source.id)
                used_tokens += token_len
                if source.id not in cleanup_ids:
                    novel_sentence_cacheables.append(source)
                if used_tokens >= final_token_budget:
                    break

            if novel_sentence_cacheables:
                first = novel_sentence_cacheables[0]
                region_cacheable = CacheableChunk(
                    id=f"{region['cacheable'].id}::support_cleanup",
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
                selected_cacheables.append((region["chunk_idx"], region_cacheable))

            if used_tokens >= final_token_budget:
                break

        return selected_cacheables

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            final_token_budget = self._resolve_final_token_budget(
                docs, self.final_token_budget
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            sentence_score_cache = {}
            scored_regions = [
                (
                    self._score_sliding_region(
                        query_vector, region, sentence_score_cache
                    ),
                    region,
                )
                for region in regions
            ]
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            cleanup_ids = self._cleanup_sentence_ids(scored_regions)

            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            if final_token_budget is None:
                for cacheable in self._select_sliding_regions_with_cleanup(
                    scored_regions,
                    cleanup_ids,
                    final_token_budget,
                ):
                    doc_id = str(cacheable.parent_doc_id)
                    for chunk_idx, doc in enumerate(docs):
                        if str(doc.id) == doc_id:
                            selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
                            break
            else:
                for chunk_idx, cacheable in self._select_sliding_regions_with_cleanup(
                    scored_regions,
                    cleanup_ids,
                    final_token_budget,
                ):
                    selected_by_doc.setdefault(chunk_idx, []).append(cacheable)

            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class FullWindowRegionColBERTSummarizer(SlidingRegionColBERTWindowSummarizer):
    def _sliding_regions_for_doc(self, doc: RetrievableChunk, chunk_idx: int):
        cacheables = [
            cacheable
            for cacheable in getattr(doc, "cacheables", []) or []
            if cacheable.text
        ]
        if not cacheables:
            return []
        specs = self.encoder.build_centered_windows(
            sentences=[cacheable.text for cacheable in cacheables],
            token_budget=self.region_token_budget,
        )
        vectors = self.encoder.encode_full_windows(specs)
        regions = []
        for center_idx, spec in enumerate(specs):
            selected_indices = tuple(
                idx for idx in spec.selected_indices if 0 <= idx < len(cacheables)
            )
            if not selected_indices:
                continue
            sentence_ids = [cacheables[idx].id for idx in selected_indices]
            cacheable = CacheableChunk(
                id=f"{doc.id}::full_window_region_{center_idx}",
                text=spec.text,
                parent_doc_id=doc.id,
                chunk_size=self.region_token_budget,
                sentence_ids=sentence_ids,
                sentence_texts=[cacheables[idx].text for idx in selected_indices],
            )
            regions.append(
                {
                    "chunk_idx": chunk_idx,
                    "center_idx": center_idx,
                    "cacheable": cacheable,
                    "vectors": (
                        vectors[center_idx]
                        if center_idx < len(vectors)
                        else torch.empty((0, 0))
                    ),
                    "selected_indices": selected_indices,
                    "source_cacheables": cacheables,
                }
            )
        return regions


class PairGainColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.pair_gamma = float(os.getenv("COLBERT_PAIR_GAIN_GAMMA", "0.2"))
        self.pair_top_m = int(os.getenv("COLBERT_PAIR_TOP_M", "16"))
        if self.pair_top_m <= 0:
            raise ValueError(
                f"COLBERT_PAIR_TOP_M must be positive, got {self.pair_top_m}"
            )

    @staticmethod
    def _pair_score(
        query_vector: torch.Tensor,
        left_vectors: torch.Tensor,
        right_vectors: torch.Tensor,
    ) -> float:
        if left_vectors.numel() == 0:
            return score_maxsim(query_vector, right_vectors)
        if right_vectors.numel() == 0:
            return score_maxsim(query_vector, left_vectors)
        return score_maxsim(
            query_vector, torch.cat([left_vectors, right_vectors], dim=0)
        )

    def _rerank_with_pair_gain(self, scored_candidates, query_vector: torch.Tensor):
        if len(scored_candidates) <= 1:
            return [
                (score, score, 0.0, candidate) for score, candidate in scored_candidates
            ]

        partner_pool = scored_candidates[: min(self.pair_top_m, len(scored_candidates))]
        reranked = []
        for center_score, candidate in scored_candidates:
            best_gain = 0.0
            for partner_score, partner in partner_pool:
                if partner is candidate:
                    continue
                if candidate.get("cacheable_id") and candidate.get(
                    "cacheable_id"
                ) == partner.get("cacheable_id"):
                    continue
                pair_score = self._pair_score(
                    query_vector, candidate["vectors"], partner["vectors"]
                )
                gain = pair_score - max(float(center_score), float(partner_score))
                if gain > best_gain:
                    best_gain = gain
            final_score = float(center_score) + self.pair_gamma * best_gain
            reranked.append((final_score, float(center_score), best_gain, candidate))
        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked

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
            keep_count = global_top_count(len(scored_candidates), self.global_top_r)
            if keep_count == 0:
                summarized_batches.append(summarized_docs)
                continue

            reranked = self._rerank_with_pair_gain(scored_candidates, query_vector)
            selected_by_doc: dict[int, list[int]] = {}
            for _, _, _, candidate in reranked[:keep_count]:
                selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                    candidate["cacheable_idx"]
                )

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class RegionPairGainColBERTWindowSummarizer(PairGainColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.region_top_r = GlobalComparisonSummarizer._parse_global_top_r(
            os.getenv("COLBERT_REGION_TOP_R", os.getenv("GLOBAL_TOP_R", "0.1"))
        )
        self.pair_scope = os.getenv("COLBERT_PAIR_SCOPE", "pool").strip().lower()
        if self.pair_scope not in {"pool", "window", "document"}:
            raise ValueError(
                "COLBERT_PAIR_SCOPE must be one of {'pool', 'window', 'document'}, "
                f"got {self.pair_scope!r}"
            )

    @staticmethod
    def _concat_candidate_vectors(candidates) -> torch.Tensor:
        vectors = [
            candidate["vectors"]
            for candidate in candidates
            if candidate["vectors"].numel() > 0
        ]
        if not vectors:
            dim = 0
            for candidate in candidates:
                if candidate["vectors"].dim() == 2:
                    dim = candidate["vectors"].shape[1]
                    break
            return torch.empty((0, dim), dtype=torch.float16)
        return torch.cat(vectors, dim=0)

    def _candidate_pool_from_top_regions(
        self, scored_candidates, query_vector: torch.Tensor
    ):
        if not scored_candidates:
            return []

        candidate_by_id = {
            candidate["cacheable_id"]: (center_score, candidate)
            for center_score, candidate in scored_candidates
            if candidate.get("cacheable_id")
        }
        region_scores = []
        for rank, (center_score, candidate) in enumerate(scored_candidates):
            region_candidates = [
                candidate_by_id[cacheable_id][1]
                for cacheable_id in candidate.get("window_cacheable_ids", [])
                if cacheable_id in candidate_by_id
            ]
            if not region_candidates:
                region_candidates = [candidate]
            region_vectors = self._concat_candidate_vectors(region_candidates)
            region_score = score_maxsim(query_vector, region_vectors)
            region_scores.append((region_score, rank, candidate, region_candidates))

        region_scores.sort(key=lambda item: (-item[0], item[1]))
        region_count = global_top_count(len(region_scores), self.region_top_r)
        selected_ids = set()
        for _, _, _, region_candidates in region_scores[:region_count]:
            for region_candidate in region_candidates:
                cacheable_id = region_candidate.get("cacheable_id")
                if cacheable_id:
                    selected_ids.add(cacheable_id)

        return [
            (center_score, candidate)
            for center_score, candidate in scored_candidates
            if candidate.get("cacheable_id") in selected_ids
        ]

    @staticmethod
    def _parent_id_from_candidate(candidate) -> str:
        cacheable_id = str(candidate.get("cacheable_id") or "")
        if "::" in cacheable_id:
            return cacheable_id.split("::", 1)[0]
        return cacheable_id.rsplit("-", 1)[0]

    def _partners_for_candidate(self, candidate, pool_candidates):
        if self.pair_scope == "pool":
            return pool_candidates
        if self.pair_scope == "window":
            window_ids = set(candidate.get("window_cacheable_ids") or [])
            return [
                item
                for item in pool_candidates
                if item[1].get("cacheable_id") in window_ids
            ]
        parent_id = self._parent_id_from_candidate(candidate)
        return [
            item
            for item in pool_candidates
            if self._parent_id_from_candidate(item[1]) == parent_id
        ]

    def _rerank_pool_with_pair_gain(self, pool_candidates, query_vector: torch.Tensor):
        if len(pool_candidates) <= 1:
            return [
                (score, score, 0.0, candidate) for score, candidate in pool_candidates
            ]

        reranked = []
        for center_score, candidate in pool_candidates:
            best_gain = 0.0
            for partner_score, partner in self._partners_for_candidate(
                candidate, pool_candidates
            ):
                if partner is candidate:
                    continue
                pair_score = self._pair_score(
                    query_vector, candidate["vectors"], partner["vectors"]
                )
                gain = pair_score - max(float(center_score), float(partner_score))
                if gain > best_gain:
                    best_gain = gain
            final_score = float(center_score) + self.pair_gamma * best_gain
            reranked.append((final_score, float(center_score), best_gain, candidate))
        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            scored_candidates = [
                (score_maxsim(query_vector, candidate["vectors"]), candidate)
                for candidate in self._iter_candidates(docs)
            ]
            scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
            keep_count = global_top_count(len(scored_candidates), self.global_top_r)
            if keep_count == 0:
                summarized_batches.append(summarized_docs)
                continue

            pool_candidates = self._candidate_pool_from_top_regions(
                scored_candidates, query_vector
            )
            if not pool_candidates:
                pool_candidates = scored_candidates
            keep_count = min(keep_count, len(pool_candidates))
            reranked = self._rerank_pool_with_pair_gain(pool_candidates, query_vector)

            selected_by_doc: dict[int, list[int]] = {}
            for _, _, _, candidate in reranked[:keep_count]:
                selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                    candidate["cacheable_idx"]
                )

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class RegionFirstColBERTWindowSummarizer(RegionPairGainColBERTWindowSummarizer):
    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            scored_candidates = [
                (score_maxsim(query_vector, candidate["vectors"]), candidate)
                for candidate in self._iter_candidates(docs)
            ]
            scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
            keep_count = global_top_count(len(scored_candidates), self.global_top_r)
            if keep_count == 0:
                summarized_batches.append(summarized_docs)
                continue

            pool_candidates = self._candidate_pool_from_top_regions(
                scored_candidates, query_vector
            )
            if not pool_candidates:
                pool_candidates = scored_candidates
            pool_candidates.sort(key=lambda item: item[0], reverse=True)
            keep_count = min(keep_count, len(pool_candidates))

            selected_by_doc: dict[int, list[int]] = {}
            for _, candidate in pool_candidates[:keep_count]:
                selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                    candidate["cacheable_idx"]
                )

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_docs[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_docs)

        return summarized_batches


class ParentAwareColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.parent_top_m = int(os.getenv("COLBERT_PARENT_TOP_M", "10"))
        if self.parent_top_m <= 0:
            raise ValueError(
                f"COLBERT_PARENT_TOP_M must be positive, got {self.parent_top_m}"
            )
        self.parent_score_mode = os.getenv("COLBERT_PARENT_SCORE_MODE", "max")
        if self.parent_score_mode not in {"max", "top2_sum", "top3_sum", "mean_top3"}:
            raise ValueError(
                "COLBERT_PARENT_SCORE_MODE must be one of "
                "{'max', 'top2_sum', 'top3_sum', 'mean_top3'}, "
                f"got {self.parent_score_mode!r}"
            )

    @staticmethod
    def _parent_id(doc: RetrievableChunk, cacheable) -> str:
        if getattr(cacheable, "parent_doc_id", None):
            return str(cacheable.parent_doc_id)
        metadata = getattr(doc, "metadata", None) or {}
        if metadata.get("parent_doc_id"):
            return str(metadata["parent_doc_id"])
        doc_id = str(getattr(doc, "id", ""))
        if "::" in doc_id:
            return doc_id.split("::", 1)[0]
        return doc_id.rsplit("-", 1)[0]

    def _parent_score(self, scores: list[float]) -> float:
        ranked = sorted(scores, reverse=True)
        if not ranked:
            return float("-inf")
        if self.parent_score_mode == "max":
            return ranked[0]
        if self.parent_score_mode == "top2_sum":
            return sum(ranked[:2])
        if self.parent_score_mode == "top3_sum":
            return sum(ranked[:3])
        if self.parent_score_mode == "mean_top3":
            return sum(ranked[:3]) / min(3, len(ranked))
        raise RuntimeError(f"unexpected parent score mode: {self.parent_score_mode}")

    def _select_parent_aware(self, docs: List[RetrievableChunk], query_vector):
        summarized_docs = [self._build_unselected_document(doc) for doc in docs]
        scored_candidates = [
            (score_maxsim(query_vector, candidate["vectors"]), candidate)
            for candidate in self._iter_candidates(docs)
        ]
        scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
        keep_count = global_top_count(len(scored_candidates), self.global_top_r)
        if keep_count == 0:
            return summarized_docs

        parent_scores: dict[str, list[float]] = {}
        parent_first_rank: dict[str, int] = {}
        for rank, (score, candidate) in enumerate(scored_candidates):
            cacheables = getattr(docs[candidate["chunk_idx"]], "cacheables", []) or []
            cacheable = cacheables[candidate["cacheable_idx"]]
            parent_id = self._parent_id(docs[candidate["chunk_idx"]], cacheable)
            parent_scores.setdefault(parent_id, []).append(float(score))
            parent_first_rank.setdefault(parent_id, rank)

        selected_parents = {
            parent_id
            for parent_id, _ in sorted(
                parent_scores.items(),
                key=lambda item: (
                    -self._parent_score(item[1]),
                    parent_first_rank[item[0]],
                ),
            )[: self.parent_top_m]
        }
        filtered_candidates = []
        for score, candidate in scored_candidates:
            cacheables = getattr(docs[candidate["chunk_idx"]], "cacheables", []) or []
            cacheable = cacheables[candidate["cacheable_idx"]]
            parent_id = self._parent_id(docs[candidate["chunk_idx"]], cacheable)
            if parent_id in selected_parents:
                filtered_candidates.append((score, candidate))

        selected_by_doc: dict[int, list[int]] = {}
        for _, candidate in filtered_candidates[:keep_count]:
            selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                candidate["cacheable_idx"]
            )

        for doc_idx, selected_indices in selected_by_doc.items():
            summarized_docs[doc_idx] = self._build_selected_document(
                docs[doc_idx], selected_indices
            )
        return summarized_docs

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        return [
            self._select_parent_aware(docs, query_vector)
            for docs, query_vector in zip(batch_top_k_docs, query_vectors)
        ]


class ParentPriorColBERTWindowSummarizer(ColBERTWindowSummarizer):
    def __init__(self):
        super().__init__()
        self.parent_prior_lambda = float(
            os.getenv("COLBERT_PARENT_PRIOR_LAMBDA", "0.2")
        )
        self.parent_prior_top_k = int(os.getenv("COLBERT_PARENT_PRIOR_TOP_K", "3"))
        if self.parent_prior_top_k <= 0:
            raise ValueError(
                f"COLBERT_PARENT_PRIOR_TOP_K must be positive, got {self.parent_prior_top_k}"
            )

    @staticmethod
    def _parent_id(doc: RetrievableChunk, cacheable) -> str:
        return ParentAwareColBERTWindowSummarizer._parent_id(doc, cacheable)

    def _parent_prior(self, scores: list[float]) -> float:
        ranked = sorted(scores, reverse=True)
        return sum(ranked[: self.parent_prior_top_k]) / min(
            self.parent_prior_top_k, len(ranked)
        )

    def _select_with_parent_prior(self, docs: List[RetrievableChunk], query_vector):
        summarized_docs = [self._build_unselected_document(doc) for doc in docs]
        scored_candidates = [
            (score_maxsim(query_vector, candidate["vectors"]), candidate)
            for candidate in self._iter_candidates(docs)
        ]
        scored_candidates = self._deduplicate_scored_candidates(scored_candidates)
        keep_count = global_top_count(len(scored_candidates), self.global_top_r)
        if keep_count == 0:
            return summarized_docs

        parent_scores: dict[str, list[float]] = {}
        candidate_parents: list[str] = []
        for score, candidate in scored_candidates:
            cacheables = getattr(docs[candidate["chunk_idx"]], "cacheables", []) or []
            cacheable = cacheables[candidate["cacheable_idx"]]
            parent_id = self._parent_id(docs[candidate["chunk_idx"]], cacheable)
            candidate_parents.append(parent_id)
            parent_scores.setdefault(parent_id, []).append(float(score))

        parent_priors = {
            parent_id: self._parent_prior(scores)
            for parent_id, scores in parent_scores.items()
        }
        reranked = []
        for rank, ((score, candidate), parent_id) in enumerate(
            zip(scored_candidates, candidate_parents)
        ):
            adjusted_score = (
                float(score) + self.parent_prior_lambda * parent_priors[parent_id]
            )
            reranked.append((adjusted_score, rank, candidate))
        reranked.sort(key=lambda item: (-item[0], item[1]))

        selected_by_doc: dict[int, list[int]] = {}
        for _, _, candidate in reranked[:keep_count]:
            selected_by_doc.setdefault(candidate["chunk_idx"], []).append(
                candidate["cacheable_idx"]
            )

        for doc_idx, selected_indices in selected_by_doc.items():
            summarized_docs[doc_idx] = self._build_selected_document(
                docs[doc_idx], selected_indices
            )
        return summarized_docs

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        query_vectors = self.encoder.encode_queries(batch_queries)
        return [
            self._select_with_parent_prior(docs, query_vector)
            for docs, query_vector in zip(batch_top_k_docs, query_vectors)
        ]
