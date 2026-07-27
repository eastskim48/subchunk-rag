"""Query-aligned gold-evidence context source for oracle reader evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List

from chunk import CacheableChunk, RetrievableChunk
from evidence_coverage import load_text_evidence_labels
from vectordb import VectorDB


class GoldEvidenceVectorDB(VectorDB):
    """Return labeled evidence passages instead of performing retrieval."""

    def __init__(self, evidence_file: str | Path):
        self.evidence_file = Path(evidence_file)
        self.labels_by_query = load_text_evidence_labels(self.evidence_file)
        self.last_find_timings: dict[str, float] = {}

    def find_top_k_docs(
        self, top_k: int, queries: List[str]
    ) -> List[List[RetrievableChunk]]:
        del top_k
        batch_docs = []
        for query in queries:
            try:
                label = self.labels_by_query[query]
            except KeyError as exc:
                raise KeyError(
                    f"gold evidence is missing for query {query!r} in "
                    f"{self.evidence_file}"
                ) from exc

            unique_passages = []
            seen_texts = set()
            for passage_id, text in zip(
                label["evidence_passage_ids"], label["evidence_texts"]
            ):
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                unique_passages.append(
                    RetrievableChunk(
                        id=f"gold-evidence::{passage_id}",
                        text=text,
                        cacheables=[
                            CacheableChunk(
                                id=f"gold-evidence::{passage_id}",
                                text=text,
                            )
                        ],
                        cache_unit="gold_evidence",
                    )
                )
            batch_docs.append(unique_passages)

        self.last_find_timings = {
            "query_time": 0.0,
            "postprocess_time": 0.0,
            "cacheable_deserialize_time": 0.0,
        }
        return batch_docs

    def store(self, chunks: List[RetrievableChunk]):
        del chunks
        raise NotImplementedError("GoldEvidenceVectorDB is read-only")
