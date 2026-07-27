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
from compressor.methods.dense import QUERY_PREFIX, DenseCompressor
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


if __name__ == "__main__":
    unittest.main()
