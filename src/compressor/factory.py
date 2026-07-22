from typing import List
from chunk import RetrievableChunk

from compressor.comparison_compressor import (
    DenseSummarizer,
    FrontCompressor,
    Summarizer,
)
from compressor.colbert_window_compressor import (
    ColBERTWindowSummarizer,
    ColBERTRerankAndRegionSummarizer,
    ColBERTRerankSummarizer,
    BudgetColBERTWindowSummarizer,
    FixedChunkColBERTRerankSummarizer,
    SlidingRegionColBERTWindowSummarizer,
)
from compressor.ml_compressor import EXITCompressor, ProvenceCompressor

compressor = None
compressor_warmed = False


COMPRESSOR_TYPES = {
    "summ": Summarizer,
    "dense": DenseSummarizer,
    "colbert_subchunk": ColBERTWindowSummarizer,
    "colbert_rerank": ColBERTRerankSummarizer,
    "colbert_chunk_rerank": FixedChunkColBERTRerankSummarizer,
    "colbert_window_budget": BudgetColBERTWindowSummarizer,
    "colbert_sliding_region": SlidingRegionColBERTWindowSummarizer,
    "rerank_and_region": ColBERTRerankAndRegionSummarizer,
    "front": FrontCompressor,
    "exit": EXITCompressor,
    "provence": ProvenceCompressor,
}


def _ensure_compressor(option):
    global compressor, compressor_warmed
    compressor_type = COMPRESSOR_TYPES.get(option)
    if compressor_type is None:
        return None
    if compressor is None or not isinstance(compressor, compressor_type):
        compressor = compressor_type()
        compressor_warmed = False
    if not compressor_warmed:
        if hasattr(compressor, "warmup_query_encoder"):
            warmup_time = compressor.warmup_query_encoder()
            print(
                f"ColBERT query encoder warmup completed in {warmup_time:.4f} seconds"
            )
        compressor_warmed = True
    return compressor


def initialize_compressor(option=None):
    if option is None:
        return None
    return _ensure_compressor(option)


def compress_docs(
    batch_queries, batch_top_k_docs, option=None
) -> List[List[RetrievableChunk]]:
    if option is None:
        return batch_top_k_docs
    active_compressor = _ensure_compressor(option)
    if active_compressor is None:
        raise ValueError(f"Unknown compression option: {option}")
    if hasattr(active_compressor, "clear_inter_batch_cache"):
        active_compressor.clear_inter_batch_cache()
    return active_compressor.compress_batch_top_k_docs(
        batch_top_k_docs=batch_top_k_docs, batch_queries=batch_queries
    )
