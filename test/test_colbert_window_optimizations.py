import json
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk, RetrievableChunk
from compressor.comparison_compressor import DenseSlidingRegionSummarizer
from compressor.colbert_window_compressor import (
    SlidingRegionColBERTWindowSummarizer,
    SupportCleanupSlidingRegionColBERTWindowSummarizer,
    _infer_configured_retrieval_chunk_size,
)
from materialize.colbert_window import (
    ARTIFACT_FORMAT,
    ColBERTWindowArtifact,
    score_maxsim,
)


class ColBERTWindowOptimizationTest(unittest.TestCase):
    def test_artifact_reuses_parent_doc_lookup_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            docs_dir = artifact_dir / "docs"
            docs_dir.mkdir()
            vectors = [
                torch.randn(2, 4, dtype=torch.float16),
                torch.randn(3, 4, dtype=torch.float16),
            ]
            torch.save(
                {
                    "format": f"{ARTIFACT_FORMAT}_doc",
                    "doc_id": "doc0.txt",
                    "cacheable_ids": ["doc0.txt::sent_0", "doc0.txt::sent_1"],
                    "center_token_vectors": vectors,
                    "window_selected_indices": [[0], [0, 1]],
                    "embedding_dim": 4,
                },
                docs_dir / "doc0.pt",
            )
            (artifact_dir / "index.json").write_text(
                json.dumps(
                    {
                        "format": ARTIFACT_FORMAT,
                        "embedding_dim": 4,
                        "docs": {"doc0.txt": {"file": "docs/doc0.pt"}},
                    }
                ),
                encoding="utf-8",
            )
            artifact = ColBERTWindowArtifact(artifact_dir)

            first_vectors = artifact.vector_lookup_for_doc_id("doc0.txt")
            second_vectors = artifact.vector_lookup_for_doc_id("doc0.txt")
            first_windows = artifact.window_ids_lookup_for_doc_id("doc0.txt")
            second_windows = artifact.window_ids_lookup_for_doc_id("doc0.txt")

        self.assertIs(first_vectors, second_vectors)
        self.assertIs(first_windows, second_windows)
        self.assertEqual(
            first_windows["doc0.txt::sent_1"], ["doc0.txt::sent_0", "doc0.txt::sent_1"]
        )

    def test_vectorized_sliding_region_scores_match_sentence_cache_path(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        query_vector = torch.randn(5, 4)
        cacheables = [
            CacheableChunk(id="doc0.txt::sent_0", text="alpha"),
            CacheableChunk(id="doc0.txt::sent_1", text="beta"),
            CacheableChunk(id="doc0.txt::sent_2", text="gamma"),
        ]
        source_vectors = [
            torch.randn(2, 4, dtype=torch.float16),
            torch.randn(4, 4, dtype=torch.float16),
            torch.randn(3, 4, dtype=torch.float16),
        ]
        regions = [
            {
                "chunk_idx": 0,
                "selected_indices": (0, 1),
                "source_cacheables": cacheables,
                "source_vectors": source_vectors,
            },
            {
                "chunk_idx": 0,
                "selected_indices": (1, 2),
                "source_cacheables": cacheables,
                "source_vectors": source_vectors,
            },
            {
                "chunk_idx": 1,
                "selected_indices": (0, 2),
                "source_cacheables": cacheables,
                "source_vectors": source_vectors,
            },
        ]

        sentence_score_cache = {}
        summarizer._populate_sentence_score_cache(
            query_vector, regions, sentence_score_cache
        )
        old_scores = [
            summarizer._score_sliding_region(query_vector, region, sentence_score_cache)
            for region in regions
        ]
        new_scores = summarizer._score_sliding_regions_vectorized(query_vector, regions)

        self.assertEqual(len(old_scores), len(new_scores))
        for old_score, new_score in zip(old_scores, new_scores):
            self.assertAlmostEqual(old_score, new_score, places=5)

    def test_vectorized_region_score_matches_concatenated_maxsim(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        query_vector = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        source_vectors = [
            torch.tensor([[0.8, 0.1], [0.1, 0.2]], dtype=torch.float16),
            torch.tensor([[0.2, 0.9]], dtype=torch.float16),
            torch.tensor([[0.5, 0.5], [0.1, 1.2]], dtype=torch.float16),
        ]
        cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha"),
            CacheableChunk(id="doc0::sent_1", text="beta"),
            CacheableChunk(id="doc0::sent_2", text="gamma"),
        ]
        regions = [
            {
                "chunk_idx": 0,
                "selected_indices": (0, 1),
                "source_cacheables": cacheables,
                "source_vectors": source_vectors,
            },
            {
                "chunk_idx": 0,
                "selected_indices": (1, 2),
                "source_cacheables": cacheables,
                "source_vectors": source_vectors,
            },
        ]

        scores = summarizer._score_sliding_regions_vectorized(query_vector, regions)

        expected_scores = [
            score_maxsim(
                query_vector, torch.cat([source_vectors[0], source_vectors[1]])
            ),
            score_maxsim(
                query_vector, torch.cat([source_vectors[1], source_vectors[2]])
            ),
        ]
        for score, expected_score in zip(scores, expected_scores):
            self.assertAlmostEqual(score, expected_score, places=5)

    def test_exhaustive_small_region_scores_match_concatenated_maxsim(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        query_vector = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
        )

        for sentence_lengths in itertools.product([0, 1, 2], repeat=3):
            source_vectors = []
            for sentence_idx, length in enumerate(sentence_lengths):
                rows = [
                    [float(sentence_idx + 1), float(token_idx + 1)]
                    for token_idx in range(length)
                ]
                source_vectors.append(torch.tensor(rows, dtype=torch.float16))
            cacheables = [
                CacheableChunk(id=f"doc0::sent_{idx}", text=f"sent {idx}")
                for idx in range(3)
            ]
            regions = []
            expected_scores = []
            for start in range(3):
                for end in range(start, 3):
                    selected_indices = tuple(range(start, end + 1))
                    regions.append(
                        {
                            "chunk_idx": 0,
                            "selected_indices": selected_indices,
                            "source_cacheables": cacheables,
                            "source_vectors": source_vectors,
                        }
                    )
                    nonempty_vectors = [
                        source_vectors[idx]
                        for idx in selected_indices
                        if source_vectors[idx].numel() > 0
                    ]
                    if nonempty_vectors:
                        expected_scores.append(
                            score_maxsim(query_vector, torch.cat(nonempty_vectors))
                        )
                    else:
                        expected_scores.append(float("-inf"))

            with self.subTest(sentence_lengths=sentence_lengths):
                vectorized_scores = summarizer._score_sliding_regions_vectorized(
                    query_vector, regions
                )
                sentence_score_cache = {}
                summarizer._populate_sentence_score_cache(
                    query_vector, regions, sentence_score_cache
                )
                cached_scores = [
                    summarizer._score_sliding_region(
                        query_vector, region, sentence_score_cache
                    )
                    for region in regions
                ]
                for score, expected_score in zip(vectorized_scores, expected_scores):
                    if expected_score == float("-inf"):
                        self.assertEqual(score, float("-inf"))
                    else:
                        self.assertAlmostEqual(score, expected_score, places=5)
                for score, expected_score in zip(cached_scores, expected_scores):
                    if expected_score == float("-inf"):
                        self.assertEqual(score, float("-inf"))
                    else:
                        self.assertAlmostEqual(score, expected_score, places=5)

    def test_sentence_score_cache_does_not_collide_on_duplicate_cacheable_ids(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        query_vector = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        first_vectors = [torch.tensor([[10.0, 0.0]], dtype=torch.float16)]
        second_vectors = [torch.tensor([[0.0, 7.0]], dtype=torch.float16)]
        first_cacheables = [CacheableChunk(id="duplicate-sent", text="alpha")]
        second_cacheables = [CacheableChunk(id="duplicate-sent", text="beta")]
        regions = [
            {
                "chunk_idx": 0,
                "selected_indices": (0,),
                "source_cacheables": first_cacheables,
                "source_vectors": first_vectors,
            },
            {
                "chunk_idx": 1,
                "selected_indices": (0,),
                "source_cacheables": second_cacheables,
                "source_vectors": second_vectors,
            },
        ]

        vectorized_scores = summarizer._score_sliding_regions_vectorized(
            query_vector, regions
        )
        sentence_score_cache = {}
        summarizer._populate_sentence_score_cache(
            query_vector, regions, sentence_score_cache
        )
        cached_scores = [
            summarizer._score_sliding_region(query_vector, region, sentence_score_cache)
            for region in regions
        ]
        expected_scores = [
            score_maxsim(query_vector, first_vectors[0]),
            score_maxsim(query_vector, second_vectors[0]),
        ]

        for score, expected_score in zip(vectorized_scores, expected_scores):
            self.assertAlmostEqual(score, expected_score, places=5)
        for score, expected_score in zip(cached_scores, expected_scores):
            self.assertAlmostEqual(score, expected_score, places=5)

    def test_region_cacheable_uses_first_and_last_selected_span_boundaries(self):
        cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
        ]
        region = SlidingRegionColBERTWindowSummarizer._make_region_cacheable(
            doc=type("Doc", (), {"id": "doc0"})(),
            center_idx=1,
            selected_indices=(1, 2),
            cacheables=cacheables,
            region_token_budget=40,
        )

        self.assertEqual(region.text, "beta gamma")
        self.assertEqual(region.sentence_ids, ["doc0::sent_1", "doc0::sent_2"])
        self.assertEqual(region.chunk_start, 6)
        self.assertEqual(region.chunk_end, 16)

    def test_exhaustive_contiguous_region_cacheable_span_boundaries(self):
        cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
            CacheableChunk(
                id="doc0::sent_3", text="delta", chunk_start=17, chunk_end=22
            ),
        ]

        for start in range(len(cacheables)):
            for end in range(start, len(cacheables)):
                selected_indices = tuple(range(start, end + 1))
                with self.subTest(selected_indices=selected_indices):
                    region = (
                        SlidingRegionColBERTWindowSummarizer._make_region_cacheable(
                            doc=type("Doc", (), {"id": "doc0"})(),
                            center_idx=start,
                            selected_indices=selected_indices,
                            cacheables=cacheables,
                            region_token_budget=40,
                        )
                    )

                    selected_cacheables = list(cacheables[start : end + 1])
                    self.assertEqual(
                        region.text,
                        " ".join(cacheable.text for cacheable in selected_cacheables),
                    )
                    self.assertEqual(
                        region.sentence_ids,
                        [cacheable.id for cacheable in selected_cacheables],
                    )
                    self.assertEqual(region.chunk_start, cacheables[start].chunk_start)
                    self.assertEqual(region.chunk_end, cacheables[end].chunk_end)

    def test_colbert_region_document_restores_source_order_after_score_order_selection(
        self,
    ):
        doc = RetrievableChunk(id="doc0", text="alpha beta gamma")
        selected_cacheables = [
            CacheableChunk(
                id="doc0::region_2", text="gamma", chunk_start=11, chunk_end=16
            ),
            CacheableChunk(
                id="doc0::region_0", text="alpha", chunk_start=0, chunk_end=5
            ),
            CacheableChunk(
                id="doc0::region_1", text="beta", chunk_start=6, chunk_end=10
            ),
        ]

        output_doc = SlidingRegionColBERTWindowSummarizer._build_region_document(
            doc, selected_cacheables
        )

        self.assertEqual(
            [cacheable.id for cacheable in output_doc.cacheables],
            ["doc0::region_0", "doc0::region_1", "doc0::region_2"],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_colbert_selection_splits_noncontiguous_novel_region_runs(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        summarizer.global_top_r = 1.0
        summarizer.region_token_budget = 40
        source_cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
        ]
        singleton_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_1",
            "parent_doc_id": "doc0",
            "selected_indices": (1,),
            "source_cacheables": source_cacheables,
        }
        wide_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_0_2",
            "parent_doc_id": "doc0",
            "selected_indices": (0, 1, 2),
            "source_cacheables": source_cacheables,
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, singleton_region), (9.0, wide_region)],
            final_token_budget=None,
        )
        output_doc = SlidingRegionColBERTWindowSummarizer._build_region_document(
            RetrievableChunk(id="doc0", text="alpha beta gamma"), selected
        )

        self.assertEqual(
            [cacheable.sentence_ids for cacheable in output_doc.cacheables],
            [["doc0::sent_0"], ["doc0::sent_1"], ["doc0::sent_2"]],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_colbert_budget_selection_adds_crossing_region_before_stop(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        summarizer.region_token_budget = 40
        summarizer._cacheable_token_lens = lambda cacheables: [
            cacheable.chunk_size for cacheable in cacheables
        ]
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        source_cacheables = [
            CacheableChunk(
                id="doc0::sent_0",
                text="alpha",
                chunk_size=1,
                chunk_start=0,
                chunk_end=5,
            ),
            CacheableChunk(
                id="doc0::sent_1",
                text="beta",
                chunk_size=1,
                chunk_start=6,
                chunk_end=10,
            ),
            CacheableChunk(
                id="doc0::sent_2",
                text="gamma",
                chunk_size=1,
                chunk_start=11,
                chunk_end=16,
            ),
        ]
        singleton_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_1",
            "parent_doc_id": "doc0",
            "selected_indices": (1,),
            "source_cacheables": source_cacheables,
        }
        wide_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_0_2",
            "parent_doc_id": "doc0",
            "selected_indices": (0, 1, 2),
            "source_cacheables": source_cacheables,
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, singleton_region), (9.0, wide_region)],
            final_token_budget=2,
        )
        output_doc = SlidingRegionColBERTWindowSummarizer._build_region_document(
            RetrievableChunk(id="doc0", text="alpha beta gamma"),
            [cacheable for _, cacheable in selected],
        )

        self.assertEqual(
            [cacheable.sentence_ids for cacheable in output_doc.cacheables],
            [["doc0::sent_0"], ["doc0::sent_1"], ["doc0::sent_2"]],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_colbert_budget_selection_is_nested_when_budget_increases(self):
        summarizer = object.__new__(SlidingRegionColBERTWindowSummarizer)
        summarizer.region_token_budget = 40
        summarizer._cacheable_token_lens = lambda cacheables: [
            cacheable.chunk_size for cacheable in cacheables
        ]
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        source_cacheables = [
            CacheableChunk(
                id="doc0::sent_0",
                text="alpha",
                chunk_size=1,
                chunk_start=0,
                chunk_end=5,
            ),
            CacheableChunk(
                id="doc0::sent_1",
                text="beta",
                chunk_size=1,
                chunk_start=6,
                chunk_end=10,
            ),
            CacheableChunk(
                id="doc0::sent_2",
                text="gamma",
                chunk_size=1,
                chunk_start=11,
                chunk_end=16,
            ),
        ]
        singleton_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_1",
            "parent_doc_id": "doc0",
            "selected_indices": (1,),
            "source_cacheables": source_cacheables,
        }
        wide_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_0_2",
            "parent_doc_id": "doc0",
            "selected_indices": (0, 1, 2),
            "source_cacheables": source_cacheables,
        }
        scored_regions = [(10.0, singleton_region), (9.0, wide_region)]

        small = summarizer._select_sliding_regions(
            scored_regions,
            final_token_budget=1,
        )
        large = summarizer._select_sliding_regions(
            scored_regions,
            final_token_budget=2,
        )

        small_ids = {cacheable.id for _, cacheable in small}
        large_ids = {cacheable.id for _, cacheable in large}
        self.assertTrue(small_ids.issubset(large_ids))
        self.assertEqual(small_ids, {"doc0::region_1::dedup_1_1"})
        output_doc = SlidingRegionColBERTWindowSummarizer._build_region_document(
            RetrievableChunk(id="doc0", text="alpha beta gamma"),
            [cacheable for _, cacheable in large],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_dense_region_document_restores_source_order_after_score_order_selection(
        self,
    ):
        doc = RetrievableChunk(id="doc0", text="alpha beta gamma")
        selected_cacheables = [
            CacheableChunk(
                id="doc0::region_2", text="gamma", chunk_start=11, chunk_end=16
            ),
            CacheableChunk(
                id="doc0::region_0", text="alpha", chunk_start=0, chunk_end=5
            ),
            CacheableChunk(
                id="doc0::region_1", text="beta", chunk_start=6, chunk_end=10
            ),
        ]

        output_doc = DenseSlidingRegionSummarizer._build_region_document(
            doc, selected_cacheables
        )

        self.assertEqual(
            [cacheable.id for cacheable in output_doc.cacheables],
            ["doc0::region_0", "doc0::region_1", "doc0::region_2"],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_dense_selection_splits_noncontiguous_novel_region_runs(self):
        summarizer = object.__new__(DenseSlidingRegionSummarizer)
        summarizer.region_token_budget = 40
        source_cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
        ]
        region = {
            "chunk_idx": 0,
            "cacheable": CacheableChunk(id="doc0::region_0_2", text="alpha beta gamma"),
            "selected_indices": (0, 1, 2),
            "source_cacheables": source_cacheables,
        }

        cacheables = summarizer._make_region_run_cacheables(region, (0, 2))

        self.assertEqual(
            [cacheable.sentence_ids for cacheable in cacheables],
            [["doc0::sent_0"], ["doc0::sent_2"]],
        )

    def test_cleanup_selection_splits_noncontiguous_kept_region_runs(self):
        summarizer = object.__new__(SupportCleanupSlidingRegionColBERTWindowSummarizer)
        summarizer.global_top_r = 1.0
        summarizer.region_token_budget = 40
        source_cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
        ]
        region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_0_2",
            "parent_doc_id": "doc0",
            "selected_indices": (0, 1, 2),
            "source_cacheables": source_cacheables,
        }

        selected = summarizer._select_sliding_regions_with_cleanup(
            [(10.0, region)],
            cleanup_ids={"doc0::sent_1"},
            final_token_budget=None,
        )
        output_doc = (
            SupportCleanupSlidingRegionColBERTWindowSummarizer._build_region_document(
                RetrievableChunk(id="doc0", text="alpha beta gamma"), selected
            )
        )

        self.assertEqual(
            [cacheable.sentence_ids for cacheable in output_doc.cacheables],
            [["doc0::sent_0"], ["doc0::sent_2"]],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "gamma"],
        )

    def test_configured_retrieval_chunk_size_prefers_explicit_env(self):
        old_explicit = os.environ.get("RETRIEVAL_CHUNK_SIZE")
        old_data_subdir = os.environ.get("DATA_SUBDIR")
        try:
            os.environ["RETRIEVAL_CHUNK_SIZE"] = "768"
            os.environ["DATA_SUBDIR"] = "sent-original-512"

            self.assertEqual(_infer_configured_retrieval_chunk_size(), 768)
        finally:
            if old_explicit is None:
                os.environ.pop("RETRIEVAL_CHUNK_SIZE", None)
            else:
                os.environ["RETRIEVAL_CHUNK_SIZE"] = old_explicit
            if old_data_subdir is None:
                os.environ.pop("DATA_SUBDIR", None)
            else:
                os.environ["DATA_SUBDIR"] = old_data_subdir

    def test_configured_retrieval_chunk_size_infers_data_subdir_suffix(self):
        old_explicit = os.environ.get("RETRIEVAL_CHUNK_SIZE")
        old_data_subdir = os.environ.get("DATA_SUBDIR")
        try:
            os.environ.pop("RETRIEVAL_CHUNK_SIZE", None)
            os.environ["DATA_SUBDIR"] = "sent-original-512"

            self.assertEqual(_infer_configured_retrieval_chunk_size(), 512)
        finally:
            if old_explicit is None:
                os.environ.pop("RETRIEVAL_CHUNK_SIZE", None)
            else:
                os.environ["RETRIEVAL_CHUNK_SIZE"] = old_explicit
            if old_data_subdir is None:
                os.environ.pop("DATA_SUBDIR", None)
            else:
                os.environ["DATA_SUBDIR"] = old_data_subdir


if __name__ == "__main__":
    unittest.main()
