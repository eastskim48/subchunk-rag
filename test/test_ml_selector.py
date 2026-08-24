import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from compressor.methods import ml_selector


class ProvenceCompressorCallTest(unittest.TestCase):
    def test_batch_call_uses_fixed_bergen_title_selection(self):
        compressor = ml_selector.ProvenceCompressor.__new__(
            ml_selector.ProvenceCompressor
        )
        compressor.model = MagicMock()
        compressor.model.process.return_value = {
            "pruned_context": [["Title. Evidence."]]
        }
        compressor.threshold = 0.1
        compressor.batch_size = 8
        compressor.reorder = False

        result = compressor._compress_texts_batch(
            [[["Title.", "Evidence."]]],
            ["question"],
        )

        self.assertEqual(result, [["Title. Evidence."]])
        compressor.model.process.assert_called_once_with(
            question=["question"],
            context=[["Title. Evidence."]],
            threshold=0.1,
            batch_size=8,
            always_select_title=True,
            enable_warnings=False,
            reorder=False,
        )

    def test_batch_compression_uses_retrieved_document_text(self):
        compressor = ml_selector.ProvenceCompressor.__new__(
            ml_selector.ProvenceCompressor
        )
        compressor.model = MagicMock()
        compressor.model.process.return_value = {
            "pruned_context": [["Retrieved evidence."]]
        }
        compressor.threshold = 0.1
        compressor.batch_size = 8
        compressor.reorder = False
        doc = ml_selector.RetrievableChunk(
            id="doc",
            text="Retrieved title. Retrieved evidence.",
            cacheables=[
                ml_selector.CacheableChunk(
                    id="boundary-sentence",
                    text="Boundary text outside the retrieved window.",
                )
            ],
        )

        result = compressor.compress_batch_top_k_docs([[doc]], ["question"])[0]

        self.assertEqual(result[0].cacheables[0].text, "Retrieved evidence.")
        self.assertEqual(
            compressor.model.process.call_args.kwargs["context"],
            [["Retrieved title. Retrieved evidence."]],
        )

    def test_reorder_keeps_official_top_five_with_original_document_ids(self):
        compressor = ml_selector.ProvenceCompressor.__new__(
            ml_selector.ProvenceCompressor
        )
        compressor.model = MagicMock()
        compressor.model.process.return_value = {
            "pruned_context": [[f"P{index}" for index in range(6)]],
            "reranking_score": [[0.2, 0.9, 0.1, 0.7, 0.4, 0.8]],
        }
        compressor.threshold = 0.1
        compressor.batch_size = 8
        compressor.reorder = True
        docs = [
            ml_selector.RetrievableChunk(
                id=f"doc-{index}",
                text=f"S{index}",
                cacheables=[
                    ml_selector.CacheableChunk(
                        id=f"cacheable-{index}",
                        text=f"S{index}",
                    )
                ],
            )
            for index in range(6)
        ]

        result = compressor.compress_batch_top_k_docs([docs], ["question"])[0]

        self.assertEqual(
            [doc.id for doc in result],
            ["doc-1", "doc-5", "doc-3", "doc-4", "doc-0"],
        )
        self.assertEqual(
            [doc.cacheables[0].text for doc in result],
            ["P1", "P5", "P3", "P4", "P0"],
        )
        self.assertFalse(compressor.model.process.call_args.kwargs["reorder"])


class EXITCompressorLoadTest(unittest.TestCase):
    def test_loads_fixed_nf4_model_without_post_load_to(self):
        base_model = object()
        exit_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = [[3553], [1294]]
        sentence_splitter = MagicMock()

        env = {
            "EXIT_MODEL_NAME": "test/exit-adapter",
            "EXIT_BASE_MODEL_NAME": "test/gemma-base",
            "EXIT_DEVICE": "cuda",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(ml_selector.torch.cuda, "is_available", return_value=True),
            patch.object(
                ml_selector.AutoModelForCausalLM,
                "from_pretrained",
                return_value=base_model,
            ) as load_base,
            patch.object(
                ml_selector.PeftModel,
                "from_pretrained",
                return_value=exit_model,
            ) as load_adapter,
            patch.object(
                ml_selector.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ),
            patch.object(
                ml_selector.spacy,
                "load",
                return_value=sentence_splitter,
            ) as load_sentence_splitter,
        ):
            compressor = ml_selector.EXITCompressor()

        load_base.assert_called_once()
        _, load_kwargs = load_base.call_args
        quantization_config = load_kwargs["quantization_config"]
        self.assertEqual(load_kwargs["device_map"], {"": "cuda"})
        self.assertEqual(load_kwargs["torch_dtype"], torch.float16)
        self.assertTrue(quantization_config.load_in_4bit)
        self.assertEqual(quantization_config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(quantization_config.bnb_4bit_use_double_quant)
        self.assertEqual(
            quantization_config.bnb_4bit_compute_dtype,
            torch.float16,
        )
        load_adapter.assert_called_once_with(base_model, "test/exit-adapter")
        exit_model.to.assert_not_called()
        exit_model.eval.assert_called_once_with()
        self.assertIs(compressor.model, exit_model)
        load_sentence_splitter.assert_called_once_with(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        sentence_splitter.enable_pipe.assert_called_once_with("senter")

    def test_batch_compression_splits_and_scores_retrieved_document_text(self):
        compressor = ml_selector.EXITCompressor.__new__(ml_selector.EXITCompressor)
        compressor.threshold = 0.5
        compressor.batch_size = 8
        compressor._split_document_sentences = MagicMock(
            return_value=["Retrieved first.", "Retrieved second."]
        )
        compressor._score_prompts = MagicMock(return_value=[0.9, 0.1])
        doc = ml_selector.RetrievableChunk(
            id="doc",
            text="Retrieved first. Retrieved second.",
            cacheables=[
                ml_selector.CacheableChunk(
                    id="boundary-sentence",
                    text="Boundary text outside the retrieved window.",
                )
            ],
        )

        result = compressor.compress_batch_top_k_docs([[doc]], ["question"])[0]

        compressor._split_document_sentences.assert_called_once_with(doc.text)
        prompts = compressor._score_prompts.call_args.args[0]
        self.assertEqual(len(prompts), 2)
        self.assertTrue(
            all(f"Full context: {doc.text}" in prompt for prompt in prompts)
        )
        self.assertTrue(all("Boundary text" not in prompt for prompt in prompts))
        self.assertEqual(result[0].cacheables[0].text, "Retrieved first.")

    def test_rejects_cpu_for_fixed_four_bit_load(self):
        with (
            patch.dict(os.environ, {"EXIT_DEVICE": "cpu"}, clear=False),
            patch.object(ml_selector.torch.cuda, "is_available", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                ml_selector.EXITCompressor()


if __name__ == "__main__":
    unittest.main()
