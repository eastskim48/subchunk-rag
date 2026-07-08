from typing import List
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
    global compressor
    compressor_type = COMPRESSOR_TYPES.get(option)
    if compressor_type is None:
        return None
    if compressor is None or not isinstance(compressor, compressor_type):
        compressor = compressor_type()
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
    return active_compressor.compress_batch_top_k_docs(
        batch_top_k_docs=batch_top_k_docs, batch_queries=batch_queries
    )
