import itertools
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk, RetrievableChunk
from compressor.colbert_window_compressor import SlidingRegionColBERTWindowSummarizer
from materialize.colbert_window import _centered_region_index_specs


class FakeRegionArtifact:
    def __init__(self, specs):
        self.specs = specs
        self.calls = []

    def region_specs_for_doc(self, doc, token_budget):
        self.calls.append((doc.id, token_budget))
        return self.specs


class NoSidecarArtifact:
    def region_specs_for_doc(self, doc, token_budget):
        return None


class FailingWindowEncoder:
    def build_centered_windows(self, sentences, token_budget):
        raise AssertionError("sidecar region specs should avoid query-time rebuild")


class FakeTokenCountEncoder:
    doc_token_overhead = 0

    def token_counts_without_specials(self, sentences):
        return [int(sentence) for sentence in sentences]

    def build_centered_windows(self, sentences, token_budget):
        raise AssertionError("fallback should use fixed region specs, not window specs")


class ColBERTRegionSpecTest(unittest.TestCase):
    def test_empty_chunk_has_no_region_specs(self):
        self.assertEqual(_centered_region_index_specs([], 10, 0), [])

    def test_oversized_center_is_singleton(self):
        specs = _centered_region_index_specs(
            [8, 1, 1], token_budget=10, doc_token_overhead=2
        )

        self.assertIn((0, (0,)), specs)
        self.assertNotIn((0, (0, 1)), specs)

    def test_too_long_right_neighbor_blocks_farther_right_subchunk(self):
        specs = _centered_region_index_specs(
            [10, 10, 1000, 10],
            token_budget=40,
            doc_token_overhead=0,
        )

        self.assertEqual(specs, [(0, (0, 1)), (2, (2,)), (3, (3,))])

    def test_too_long_left_neighbor_blocks_farther_left_subchunk(self):
        specs = _centered_region_index_specs(
            [10, 1000, 10, 10],
            token_budget=40,
            doc_token_overhead=0,
        )

        self.assertEqual(specs, [(0, (0,)), (1, (1,)), (2, (2, 3))])

    def test_duplicate_regions_are_removed(self):
        specs = _centered_region_index_specs(
            [10, 10, 10, 10],
            token_budget=40,
            doc_token_overhead=0,
        )

        self.assertEqual(specs, [(0, (0, 1, 2, 3))])

    def test_balanced_budget_produces_contiguous_source_order_regions(self):
        specs = _centered_region_index_specs(
            [5, 5, 5, 5, 5],
            token_budget=15,
            doc_token_overhead=0,
        )

        self.assertEqual(specs, [(0, (0, 1, 2)), (2, (1, 2, 3)), (3, (2, 3, 4))])

    def test_exhaustive_small_inputs_preserve_region_invariants(self):
        token_values = [1, 2, 5, 20]
        budgets = [3, 5, 8, 12, 25]
        overheads = [0, 2]

        for length in range(1, 7):
            for token_counts in itertools.product(token_values, repeat=length):
                for token_budget in budgets:
                    for doc_token_overhead in overheads:
                        with self.subTest(
                            token_counts=token_counts,
                            token_budget=token_budget,
                            doc_token_overhead=doc_token_overhead,
                        ):
                            specs = _centered_region_index_specs(
                                list(token_counts),
                                token_budget=token_budget,
                                doc_token_overhead=doc_token_overhead,
                            )
                            seen_regions = set()
                            for center_idx, selected_indices in specs:
                                selected = tuple(selected_indices)
                                self.assertNotIn(selected, seen_regions)
                                seen_regions.add(selected)
                                self.assertIn(center_idx, selected)
                                self.assertEqual(selected, tuple(sorted(set(selected))))
                                self.assertEqual(
                                    selected,
                                    tuple(range(selected[0], selected[-1] + 1)),
                                )
                                self.assertTrue(
                                    all(0 <= idx < length for idx in selected)
                                )

                                center_cost = (
                                    token_counts[center_idx] + doc_token_overhead
                                )
                                if center_cost >= token_budget:
                                    self.assertEqual(selected, (center_idx,))
                                else:
                                    total_cost = (
                                        sum(token_counts[idx] for idx in selected)
                                        + doc_token_overhead
                                    )
                                    self.assertLessEqual(total_cost, token_budget)

                                    left_idx = selected[0] - 1
                                    right_idx = selected[-1] + 1
                                    if left_idx >= 0:
                                        self.assertGreater(
                                            total_cost + token_counts[left_idx],
                                            token_budget,
                                        )
                                    if right_idx < length:
                                        self.assertGreater(
                                            total_cost + token_counts[right_idx],
                                            token_budget,
                                        )

    def test_sliding_region_summarizer_uses_materialized_sidecar_specs(self):
        doc = RetrievableChunk(
            id="chunk0",
            text="alpha beta gamma",
            cacheables=[
                CacheableChunk(id="chunk0::sent_0", text="alpha"),
                CacheableChunk(id="chunk0::sent_1", text="beta"),
                CacheableChunk(id="chunk0::sent_2", text="gamma"),
            ],
        )
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        summarizer.region_token_budget = 40
        summarizer._sliding_region_spec_cache = {}
        summarizer.artifact = FakeRegionArtifact([(0, (0, 1)), (2, (2,))])
        summarizer.encoder = FailingWindowEncoder()

        specs = summarizer._cached_sliding_region_specs(doc, doc.cacheables)

        self.assertEqual(specs, [(0, (0, 1)), (2, (2,))])
        self.assertEqual(summarizer.artifact.calls, [("chunk0", 40)])

    def test_sliding_region_fallback_matches_fixed_region_specs(self):
        doc = RetrievableChunk(
            id="chunk0",
            text="10 10 1000 10",
            cacheables=[
                CacheableChunk(id="chunk0::sent_0", text="10"),
                CacheableChunk(id="chunk0::sent_1", text="10"),
                CacheableChunk(id="chunk0::sent_2", text="1000"),
                CacheableChunk(id="chunk0::sent_3", text="10"),
            ],
        )
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        summarizer.region_token_budget = 40
        summarizer._sliding_region_spec_cache = {}
        summarizer.artifact = NoSidecarArtifact()
        summarizer.encoder = FakeTokenCountEncoder()

        specs = summarizer._cached_sliding_region_specs(doc, doc.cacheables)

        self.assertEqual(specs, [(0, (0, 1)), (2, (2,)), (3, (3,))])


if __name__ == "__main__":
    unittest.main()
