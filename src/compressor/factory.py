from typing import List
import os
from chunk import RetrievableChunk

from compressor.comparison_compressor import (
    ComparisonSummarizer,
    DenseSlidingRegionSummarizer,
    FrontCompressor,
    GlobalComparisonSummarizer,
    MaterializedBudgetComparisonSummarizer,
    MaterializedGlobalComparisonSummarizer,
    Summarizer,
    TitleRRFSummarizer,
    WindowedComparisonSummarizer,
)
from compressor.bm25_compressor import (
    BM25GlobalAsyncSummarizer,
    BM25GlobalSummarizer,
    HybridBgeBM25AsyncSummarizer,
    HybridBgeBM25RRFAsyncSummarizer,
    HybridBgeBM25RRFSummarizer,
    HybridBgeBM25Summarizer,
    MaterializedGlobalComparisonAsyncSummarizer,
)
from compressor.colbert_window_compressor import (
    ColBERTWindowSummarizer,
    BudgetColBERTWindowSummarizer,
    FixedChunkColBERTRerankSummarizer,
    FixedRegionColBERTSummarizer,
    FullWindowRegionColBERTSummarizer,
    PairGainColBERTWindowSummarizer,
    ParentAwareColBERTWindowSummarizer,
    ParentPriorColBERTWindowSummarizer,
    PrunedSlidingRegionColBERTWindowSummarizer,
    RegionFirstColBERTWindowSummarizer,
    RegionPairGainColBERTWindowSummarizer,
    SlidingRegionColBERTWindowSummarizer,
    SupportCleanupSlidingRegionColBERTWindowSummarizer,
    SupportPrunedSlidingRegionColBERTWindowSummarizer,
)
from compressor.ml_compressor import EXITCompressor, ProvenceCompressor

compressor = None
compressor_warmed = False


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    return default


COMPRESSOR_TYPES = {
    "summ": Summarizer,
    "compare": ComparisonSummarizer,
    "compare_all": GlobalComparisonSummarizer,
    "compare_all_materialized": MaterializedGlobalComparisonSummarizer,
    "compare_all_materialized_budget": MaterializedBudgetComparisonSummarizer,
    "dense_sliding_region": DenseSlidingRegionSummarizer,
    "title_rrf": TitleRRFSummarizer,
    "compare_all_materialized_async": MaterializedGlobalComparisonAsyncSummarizer,
    "compare_window": WindowedComparisonSummarizer,
    "bm25_global": BM25GlobalSummarizer,
    "bm25_global_async": BM25GlobalAsyncSummarizer,
    "hybrid_bge_bm25": HybridBgeBM25Summarizer,
    "hybrid_bge_bm25_rrf": HybridBgeBM25RRFSummarizer,
    "hybrid_bge_bm25_async": HybridBgeBM25AsyncSummarizer,
    "hybrid_bge_bm25_rrf_async": HybridBgeBM25RRFAsyncSummarizer,
    "colbert_window": ColBERTWindowSummarizer,
    "colbert_chunk_rerank": FixedChunkColBERTRerankSummarizer,
    "colbert_window_budget": BudgetColBERTWindowSummarizer,
    "colbert_fixed_region": FixedRegionColBERTSummarizer,
    "colbert_sliding_region": SlidingRegionColBERTWindowSummarizer,
    "colbert_sliding_region_pruned": PrunedSlidingRegionColBERTWindowSummarizer,
    "colbert_sliding_region_support_cleanup": SupportCleanupSlidingRegionColBERTWindowSummarizer,
    "colbert_sliding_region_support_pruned": SupportPrunedSlidingRegionColBERTWindowSummarizer,
    "colbert_full_window_region": FullWindowRegionColBERTSummarizer,
    "colbert_window_pair_gain": PairGainColBERTWindowSummarizer,
    "colbert_window_region_first": RegionFirstColBERTWindowSummarizer,
    "colbert_window_region_pair_gain": RegionPairGainColBERTWindowSummarizer,
    "colbert_window_parent": ParentAwareColBERTWindowSummarizer,
    "colbert_window_parent_prior": ParentPriorColBERTWindowSummarizer,
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
    if (
        not compressor_warmed
        and _parse_bool(os.getenv("COLBERT_WARMUP_QUERY_ENCODER"), True)
        and hasattr(compressor, "warmup_query_encoder")
    ):
        warmup_time = compressor.warmup_query_encoder()
        print(f"ColBERT query encoder warmup completed in {warmup_time:.4f} seconds")
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
    if _parse_bool(os.getenv("COLBERT_CLEAR_INTER_BATCH_CACHE"), True) and hasattr(
        active_compressor, "clear_inter_batch_cache"
    ):
        active_compressor.clear_inter_batch_cache()
    return active_compressor.compress_batch_top_k_docs(
        batch_top_k_docs=batch_top_k_docs, batch_queries=batch_queries
    )
