import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from encoder.dense import (
    BGE_SMALL_MODEL,
    BGE_SMALL_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    E5_SMALL_MODEL,
    default_passage_prefix,
    default_pooling_mode,
    default_query_prefix,
    pool_token_embeddings,
)
from vectordb import ChromaDB


class DenseEmbeddingBackendTest(unittest.TestCase):
    def test_model_specific_query_prefixes(self):
        self.assertEqual(
            default_query_prefix(BGE_SMALL_MODEL),
            BGE_SMALL_QUERY_PREFIX,
        )
        self.assertEqual(default_query_prefix(E5_SMALL_MODEL), E5_QUERY_PREFIX)
        self.assertEqual(default_query_prefix("unknown/model"), "")

    def test_model_specific_passage_prefixes(self):
        self.assertEqual(default_passage_prefix(BGE_SMALL_MODEL), "")
        self.assertEqual(
            default_passage_prefix(E5_SMALL_MODEL),
            E5_PASSAGE_PREFIX,
        )

    def test_model_specific_pooling_modes(self):
        self.assertEqual(default_pooling_mode(BGE_SMALL_MODEL), "cls")
        self.assertEqual(default_pooling_mode(E5_SMALL_MODEL), "mean")

    def test_pooling_implements_bge_cls_and_e5_masked_mean(self):
        embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])

        cls = pool_token_embeddings(embeddings, mask, "cls")
        mean = pool_token_embeddings(embeddings, mask, "mean")

        torch.testing.assert_close(cls, torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(mean, torch.tensor([[2.0, 3.0]]))

    def test_chroma_backend_names_are_bound_to_expected_models(self):
        self.assertEqual(
            ChromaDB.DEFAULT_EMBED_BACKEND,
            ChromaDB.BGE_SMALL_EMBED_BACKEND,
        )
        self.assertEqual(
            ChromaDB.DENSE_EMBED_BACKENDS["bge_small_v1_5"],
            BGE_SMALL_MODEL,
        )
        self.assertEqual(
            ChromaDB.DENSE_EMBED_BACKENDS["e5_small_v2"],
            E5_SMALL_MODEL,
        )

    def test_chroma_implicit_backend_uses_bge_small(self):
        class FakeClient:
            @staticmethod
            def get_or_create_collection(**kwargs):
                return kwargs

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("vectordb.chromadb.PersistentClient", return_value=FakeClient()),
            patch("vectordb.DenseEmbeddingFunction") as embedding_function,
        ):
            ChromaDB._get_chroma_client(
                "/tmp/not-used",
                embedding_device="cpu",
                embedding_batch_size=32,
            )

        embedding_function.assert_called_once_with(
            model_name=BGE_SMALL_MODEL,
            function_name="bge_small_v1_5",
        )

    def test_chroma_dense_backend_does_not_mutate_shared_hnsw_config(self):
        expected = copy.deepcopy(ChromaDB.DEFAULT_COLLECTION_CONFIGURATION)

        class FakeClient:
            @staticmethod
            def get_or_create_collection(**kwargs):
                kwargs["configuration"]["embedding_function"] = "mutated"
                return object()

        with (
            patch.dict(
                os.environ,
                {"CHROMA_EMBED_BACKEND": "bge_small_v1_5"},
                clear=False,
            ),
            patch("vectordb.chromadb.PersistentClient", return_value=FakeClient()),
            patch("vectordb.DenseEmbeddingFunction", return_value=object()),
        ):
            ChromaDB._get_chroma_client(
                "/tmp/not-used",
                embedding_device="cpu",
                embedding_batch_size=32,
            )

        self.assertEqual(ChromaDB.DEFAULT_COLLECTION_CONFIGURATION, expected)


if __name__ == "__main__":
    unittest.main()
