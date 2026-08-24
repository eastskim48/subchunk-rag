import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunk import CacheableChunk, RetrievableChunk
from compressor.methods.dense import (
    QUERY_PREFIX,
    DenseCompressor,
    DenseOnlineCompressor,
    DenseSlidingSubchunkCompressor,
    DenseSlidingRegionMaxCompressor,
    DenseSlidingRegionAvgCompressor,
)
from compressor.token_budget import TokenBudgetMixin
from materialize.dense_materializer import DenseEmbeddingWriter


class DenseCompressorTest(unittest.TestCase):
    def test_dense_artifact_filename_and_format_match_reader(self):
        self.assertEqual(DenseEmbeddingWriter.OUTPUT_FILENAME, "dense_embed.pt")
        payload = {
            "format": "dense_embed_single_v1",
            "embedding_model": "fake-model",
            "embedding_query_prefix": QUERY_PREFIX,
            "doc_index": {},
            "embeddings": torch.empty((0, 0), dtype=torch.float16),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            torch.save(payload, Path(output_dir) / DenseEmbeddingWriter.OUTPUT_FILENAME)
            compressor = object.__new__(DenseCompressor)
            compressor.dense_embed_dir = output_dir
            compressor.embedding_model = "fake-model"
            compressor.query_prefix = QUERY_PREFIX

            loaded = compressor._load_dense_embed_payload()

        self.assertEqual(loaded["format"], "dense_embed_single_v1")

    def test_token_budget_requires_exactly_one_controller(self):
        budget = TokenBudgetMixin()
        with patch.dict(os.environ, {"MODEL_NAME": "fake-model"}, clear=True):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                budget._initialize_token_budget()

        with patch.dict(
            os.environ,
            {
                "MODEL_NAME": "fake-model",
                "RETAIN_TOKEN_RATIO": "0.5",
                "FINAL_TOKEN_BUDGET": "100",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                budget._initialize_token_budget()

    def test_token_budget_treats_empty_controller_as_unset(self):
        with patch(
            "compressor.token_budget.AutoTokenizer.from_pretrained",
            return_value=object(),
        ):
            ratio_budget = TokenBudgetMixin()
            with patch.dict(
                os.environ,
                {
                    "MODEL_NAME": "fake-model",
                    "RETAIN_TOKEN_RATIO": "0.25",
                    "FINAL_TOKEN_BUDGET": "",
                },
                clear=True,
            ):
                ratio_budget._initialize_token_budget()
            self.assertEqual(ratio_budget.retain_token_ratio, 0.25)
            self.assertIsNone(ratio_budget.final_token_budget)

            absolute_budget = TokenBudgetMixin()
            with patch.dict(
                os.environ,
                {
                    "MODEL_NAME": "fake-model",
                    "RETAIN_TOKEN_RATIO": " ",
                    "FINAL_TOKEN_BUDGET": "600",
                },
                clear=True,
            ):
                absolute_budget._initialize_token_budget()
            self.assertIsNone(absolute_budget.retain_token_ratio)
            self.assertEqual(absolute_budget.final_token_budget, 600)

    def test_dense_selection_deduplicates_and_stops_after_reaching_budget(self):
        summarizer = object.__new__(DenseCompressor)
        summarizer.retain_token_ratio = None
        summarizer.final_token_budget = 2
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        first_doc = RetrievableChunk(
            id="first-doc",
            text="",
            cacheables=[
                CacheableChunk(id="a", text="a", chunk_size=1),
                CacheableChunk(id="shared", text="shared", chunk_size=1),
            ],
        )
        second_doc = RetrievableChunk(
            id="second-doc",
            text="",
            cacheables=[
                CacheableChunk(id="shared", text="shared", chunk_size=1),
                CacheableChunk(id="c", text="c", chunk_size=1),
            ],
        )
        summarizer._score_chunk_texts_for_batch = lambda docs, queries: (
            [[["a", "shared"], ["shared", "c"]]],
            torch.tensor([0.1, 0.9, 0.8, 0.7]),
        )

        output = summarizer.compress_batch_top_k_docs(
            [[first_doc, second_doc]], ["query"]
        )[0]

        self.assertEqual(
            [[cacheable.id for cacheable in doc.cacheables] for doc in output],
            [["shared"], ["c"]],
        )

    def test_dense_selection_adds_last_subchunk_before_budget_check(self):
        summarizer = object.__new__(DenseCompressor)
        summarizer.retain_token_ratio = None
        summarizer.final_token_budget = 4
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        doc = RetrievableChunk(
            id="doc",
            text="",
            cacheables=[
                CacheableChunk(id="a", text="a", chunk_size=3),
                CacheableChunk(id="b", text="b", chunk_size=2),
            ],
        )
        summarizer._score_chunk_texts_for_batch = lambda docs, queries: (
            [[["a", "b"]]],
            torch.tensor([0.9, 0.8]),
        )

        output = summarizer.compress_batch_top_k_docs([[doc]], ["query"])[0][0]

        self.assertEqual([cacheable.id for cacheable in output.cacheables], ["a", "b"])

    def test_dense_online_scores_runtime_candidate_embeddings(self):
        compressor = object.__new__(DenseOnlineCompressor)
        compressor.query_prefix = "query: "
        embedded_inputs = []

        def embed_texts(texts):
            embedded_inputs.append(texts)
            vectors = {
                "query: first": [1.0, 0.0],
                "query: second": [0.0, 1.0],
                "a": [0.8, 0.2],
                "b": [0.1, 0.9],
                "c": [0.4, 0.6],
            }
            return torch.tensor([vectors[text] for text in texts])

        compressor._embed_texts = embed_texts
        first_doc = RetrievableChunk(
            id="first-doc",
            text="",
            cacheables=[
                CacheableChunk(id="a", text="a", chunk_size=1),
                CacheableChunk(id="b", text="b", chunk_size=1),
            ],
        )
        second_doc = RetrievableChunk(
            id="second-doc",
            text="",
            cacheables=[CacheableChunk(id="c", text="c", chunk_size=1)],
        )

        texts, scores = compressor._score_chunk_texts_for_batch(
            [[first_doc], [second_doc]], ["first", "second"]
        )

        self.assertEqual(texts, [[["a", "b"]], [["c"]]])
        self.assertEqual(
            embedded_inputs,
            [["query: first", "query: second"], ["a", "b", "c"]],
        )
        torch.testing.assert_close(scores, torch.tensor([0.8, 0.1, 0.6]))

    def test_dense_online_deduplicates_and_caches_stable_candidate_ids(self):
        compressor = object.__new__(DenseOnlineCompressor)
        compressor.query_prefix = ""
        embedded_inputs = []

        def embed_texts(texts):
            embedded_inputs.append(texts)
            vectors = {
                "first query": [1.0, 0.0],
                "second query": [0.0, 1.0],
                "shared": [0.8, 0.2],
                "new": [0.1, 0.9],
            }
            return torch.tensor([vectors[text] for text in texts])

        compressor._embed_texts = embed_texts
        first_shared = CacheableChunk(id="shared-id", text="shared", chunk_size=1)
        second_shared = CacheableChunk(id="shared-id", text="shared", chunk_size=1)
        first_docs = [
            RetrievableChunk(id="first", text="", cacheables=[first_shared]),
            RetrievableChunk(id="second", text="", cacheables=[second_shared]),
        ]

        _, first_scores = compressor._score_chunk_texts_for_batch(
            [first_docs], ["first query"]
        )
        next_doc = RetrievableChunk(
            id="third",
            text="",
            cacheables=[
                CacheableChunk(id="shared-id", text="shared", chunk_size=1),
                CacheableChunk(id="new-id", text="new", chunk_size=1),
            ],
        )
        _, second_scores = compressor._score_chunk_texts_for_batch(
            [[next_doc]], ["second query"]
        )

        self.assertEqual(
            embedded_inputs,
            [
                ["first query"],
                ["shared"],
                ["second query"],
                ["new"],
            ],
        )
        torch.testing.assert_close(first_scores, torch.tensor([0.8, 0.8]))
        torch.testing.assert_close(second_scores, torch.tensor([0.2, 0.9]))
        self.assertEqual(compressor.runtime_cache_hits, 1)
        self.assertEqual(compressor.runtime_cache_misses, 2)

    def test_dense_online_rejects_cached_id_text_conflicts(self):
        compressor = object.__new__(DenseOnlineCompressor)
        compressor.query_prefix = ""
        compressor._embed_texts = lambda texts: torch.ones((len(texts), 2))
        first = RetrievableChunk(
            id="first",
            text="",
            cacheables=[CacheableChunk(id="shared", text="first")],
        )
        conflicting = RetrievableChunk(
            id="second",
            text="",
            cacheables=[CacheableChunk(id="shared", text="different")],
        )

        compressor._score_chunk_texts_for_batch([[first]], ["query"])
        with self.assertRaisesRegex(ValueError, "conflicting texts"):
            compressor._score_chunk_texts_for_batch([[conflicting]], ["query"])

    def test_dense_sliding_region_max_and_avg_aggregate_same_member_scores(self):
        source_cacheables = [
            CacheableChunk(id="a", text="a"),
            CacheableChunk(id="b", text="b"),
            CacheableChunk(id="c", text="c"),
            CacheableChunk(id="d", text="d"),
        ]
        regions = [
            {
                "selected_indices": (0, 1),
                "source_cacheables": source_cacheables,
            },
            {
                "selected_indices": (2, 3),
                "source_cacheables": source_cacheables,
            },
        ]
        embedded_inputs = []

        def embed_texts(texts):
            embedded_inputs.append(texts)
            vectors = {
                "query": [1.0, 0.0],
                "a": [0.9, 0.0],
                "b": [0.1, 0.0],
                "c": [0.7, 0.0],
                "d": [0.7, 0.0],
            }
            return torch.tensor([vectors[text] for text in texts])

        max_compressor = object.__new__(DenseSlidingRegionMaxCompressor)
        max_compressor.query_prefix = ""
        max_compressor._embed_texts = embed_texts
        avg_compressor = object.__new__(DenseSlidingRegionAvgCompressor)
        avg_compressor.query_prefix = ""
        avg_compressor._embed_texts = embed_texts

        max_scores = max_compressor._score_dense_regions_for_batch([regions], ["query"])
        avg_scores = avg_compressor._score_dense_regions_for_batch([regions], ["query"])

        self.assertEqual(
            embedded_inputs,
            [
                ["query"],
                ["a", "b", "c", "d"],
                ["query"],
                ["a", "b", "c", "d"],
            ],
        )
        self.assertAlmostEqual(max_scores[0][0], 0.9)
        self.assertAlmostEqual(max_scores[0][1], 0.7)
        self.assertAlmostEqual(avg_scores[0][0], 0.5)
        self.assertAlmostEqual(avg_scores[0][1], 0.7)
        self.assertGreater(max_scores[0][0], max_scores[0][1])
        self.assertLess(avg_scores[0][0], avg_scores[0][1])

    def test_dense_sliding_region_deduplicates_ids_and_rejects_conflicts(self):
        compressor = object.__new__(DenseSlidingRegionMaxCompressor)
        compressor.query_prefix = ""
        embedded_inputs = []
        compressor._embed_texts = lambda texts: (
            embedded_inputs.append(texts) or torch.ones((len(texts), 2))
        )
        first = CacheableChunk(id="shared", text="same")
        second = CacheableChunk(id="shared", text="same")
        regions = [
            {"selected_indices": (0,), "source_cacheables": [first]},
            {"selected_indices": (0,), "source_cacheables": [second]},
        ]

        compressor._score_dense_regions_for_batch([regions], ["query"])

        self.assertEqual(embedded_inputs, [["query"], ["same"]])

        conflicting = CacheableChunk(id="shared", text="different")
        conflicting_regions = [
            {"selected_indices": (0,), "source_cacheables": [first]},
            {"selected_indices": (0,), "source_cacheables": [conflicting]},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting texts"):
            compressor._score_dense_regions_for_batch([conflicting_regions], ["query"])

    def test_dense_sliding_region_reuses_candidates_across_evaluation_batches(self):
        compressor = object.__new__(DenseSlidingRegionMaxCompressor)
        compressor.query_prefix = ""
        compressor._sliding_region_spec_cache = {"temporary": object()}
        embedded_inputs = []
        compressor._embed_texts = lambda texts: (
            embedded_inputs.append(texts) or torch.ones((len(texts), 2))
        )
        cacheable = CacheableChunk(id="shared", text="same")
        regions = [{"selected_indices": (0,), "source_cacheables": [cacheable]}]

        compressor._score_dense_regions_for_batch([regions], ["query"])
        compressor.clear_inter_batch_cache()
        compressor._score_dense_regions_for_batch([regions], ["query"])

        self.assertEqual(embedded_inputs, [["query"], ["same"], ["query"]])
        self.assertEqual(compressor._sliding_region_spec_cache, {})
        self.assertIn("shared", compressor._runtime_cacheable_embeddings)
        self.assertEqual(compressor.runtime_cache_hits, 1)
        self.assertEqual(compressor.runtime_cache_misses, 1)

    def test_dense_sliding_region_warmup_uses_dense_embedder(self):
        compressor = object.__new__(DenseSlidingRegionMaxCompressor)
        compressor.query_prefix = "query: "
        embedded_inputs = []
        compressor._embed_texts = lambda texts: (
            embedded_inputs.append(texts) or torch.ones((len(texts), 2))
        )

        elapsed = compressor.warmup_query_encoder()

        self.assertEqual(embedded_inputs, [["query: warmup query"]])
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(compressor.query_encoder_warmup_time, elapsed)
        self.assertFalse(hasattr(compressor, "encoder"))

    def test_dense_sliding_subchunk_embeds_complete_region_text(self):
        compressor = object.__new__(DenseSlidingSubchunkCompressor)
        compressor.query_prefix = ""
        embedded_inputs = []

        def embed_texts(texts):
            embedded_inputs.append(texts)
            vectors = {
                "query": [1.0, 0.0],
                "first second": [0.8, 0.2],
                "second third": [0.3, 0.7],
            }
            return torch.tensor([vectors[text] for text in texts])

        compressor._embed_texts = embed_texts
        source_cacheables = [
            CacheableChunk(id="first", text="first"),
            CacheableChunk(id="second", text="second"),
            CacheableChunk(id="third", text="third"),
        ]
        regions = [
            {
                "region_id": "chunk::region_0",
                "parent_doc_id": "parent",
                "selected_indices": (0, 1),
                "source_cacheables": source_cacheables,
            },
            {
                "region_id": "chunk::region_1",
                "parent_doc_id": "parent",
                "selected_indices": (1, 2),
                "source_cacheables": source_cacheables,
            },
        ]

        scores = compressor._score_dense_regions_for_batch([regions], ["query"])

        self.assertEqual(
            embedded_inputs,
            [["query"], ["first second", "second third"]],
        )
        self.assertAlmostEqual(scores[0][0], 0.8)
        self.assertAlmostEqual(scores[0][1], 0.3)

    def test_dense_sliding_subchunk_reuses_runtime_region_embeddings(self):
        compressor = object.__new__(DenseSlidingSubchunkCompressor)
        compressor.query_prefix = ""
        embedded_inputs = []
        compressor._embed_texts = lambda texts: (
            embedded_inputs.append(texts) or torch.ones((len(texts), 2))
        )
        source_cacheables = [CacheableChunk(id="sentence", text="sentence")]
        regions = [
            {
                "region_id": "chunk::region_0",
                "parent_doc_id": "parent",
                "selected_indices": (0,),
                "source_cacheables": source_cacheables,
            }
        ]

        compressor._score_dense_regions_for_batch([regions], ["query"])
        compressor._score_dense_regions_for_batch([regions], ["query"])

        self.assertEqual(embedded_inputs, [["query"], ["sentence"], ["query"]])
        self.assertEqual(compressor.runtime_cache_hits, 1)
        self.assertEqual(compressor.runtime_cache_misses, 1)


if __name__ == "__main__":
    unittest.main()
