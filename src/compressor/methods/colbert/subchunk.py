"""Direct subchunk selection with contextualized ColBERT representations."""

from typing import List

import torch

from chunk import RetrievableChunk
from compressor.methods.colbert.base import ColBERTWindowCompressorBase
from compressor.methods.colbert.scoring import score_maxsim


class ColBERTSubchunkCompressor(ColBERTWindowCompressorBase):
    """Select individual subchunks using contextualized ColBERT vectors."""

    def _iter_candidates(self, docs: List[RetrievableChunk]):
        candidates = []
        for chunk_idx, doc in enumerate(docs):
            vectors = self.artifact.vectors_for_doc(doc)
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
            final_token_budget = self._resolve_final_token_budget(docs)
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
                selected_ids.add(cacheable_id)
                selected_candidates.append((score, candidate))
                used_tokens += self._cacheable_token_len(cacheables[cacheable_idx])
                # Selection policy includes the current candidate, then stops
                # once the accumulated prompt-visible tokens reach the budget.
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
