"""Common interface and output helpers for context compressors."""

from typing import List

from chunk import RetrievableChunk


class Compressor:
    """Base class for compressors operating on batches of retrieved chunks."""

    def __init__(self):
        pass

    def _build_selected_document(
        self, rchunk: RetrievableChunk, selected_indices: List[int]
    ) -> RetrievableChunk:
        """Clone a document while retaining selected cacheable units in source order."""

        cacheables = list(getattr(rchunk, "cacheables", None) or [])
        cloned = rchunk.clone()
        if not cacheables:
            return cloned
        limit = len(cacheables)
        valid_indices = sorted({idx for idx in selected_indices if 0 <= idx < limit})
        # Keep one unit when a selector returns no valid index for a non-empty chunk.
        if not valid_indices and limit > 0:
            valid_indices = [0]
        selected_cacheables = [cacheables[idx].clone() for idx in valid_indices]
        cloned.cacheables = selected_cacheables
        return cloned

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        for rchunks, query in zip(batch_top_k_docs, batch_queries):
            summarized_docs = []
            for rchunk in rchunks:
                cloned = rchunk.clone()
                cloned.text = self.compress(document_text=rchunk.text, query=query)
                summarized_docs.append(cloned)
            summarized_batches.append(summarized_docs)
        return summarized_batches

    def compress(self, document_text: str, query: str) -> str:
        raise NotImplementedError("Subclasses must implement the compress method.")
