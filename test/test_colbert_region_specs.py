import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk, RetrievableChunk
from colbert_artifact import ARTIFACT_FORMAT, DATA_ARTIFACT_FORMAT
from colbert_metadata import ColBERTMetadataReader, ColBERTMetadataWriter
from compressor.methods.colbert import (
    ColBERTRerankAndRegionCompressor,
    ColBERTSlidingRegionCompressor,
)
from compressor.methods.colbert.region import (
    _validate_retrieval_chunk_larger_than_region,
)
from compressor.token_budget import TokenBudgetMixin
from materialize.colbert_materializer import (
    ColBERTWindowEncoder,
    add_region_specs_to_colbert_window_data,
    _window_bounded_region_index_specs,
)
import materialize.colbert_materializer as colbert_materializer


class FakeRegionArtifact:
    def __init__(self, specs):
        self.specs = specs
        self.calls = []

    def region_specs_for_doc(self, doc, token_budget):
        self.calls.append((doc.id, token_budget))
        return self.specs


class FailingWindowEncoder:
    def build_centered_windows(self, sentences, token_budget):
        raise AssertionError("sidecar region specs should avoid query-time rebuild")


class CountingBudgetTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        del kwargs
        self.calls.append(list(texts))
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class ColBERTRegionSpecTest(unittest.TestCase):
    def test_retrieval_chunk_only_needs_to_exceed_region_budget(self):
        _validate_retrieval_chunk_larger_than_region(256, 180)

        with self.assertRaisesRegex(ValueError, "retrieval chunk size must be larger"):
            _validate_retrieval_chunk_larger_than_region(180, 180)

    def test_budget_tokenizer_uses_eval_model_name(self):
        budget = TokenBudgetMixin()
        with (
            patch.dict(
                "os.environ",
                {
                    "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct",
                    "RETAIN_TOKEN_RATIO": "0.5",
                },
                clear=True,
            ),
            patch(
                "compressor.token_budget.AutoTokenizer.from_pretrained",
                return_value=CountingBudgetTokenizer(),
            ),
        ):
            budget._initialize_token_budget()

        self.assertEqual(
            budget.budget_tokenizer_name,
            "meta-llama/Llama-3.2-1B-Instruct",
        )

    def test_budget_tokenizer_requires_eval_model_name(self):
        budget = TokenBudgetMixin()
        with patch.dict("os.environ", {"RETAIN_TOKEN_RATIO": "0.5"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MODEL_NAME must be set"):
                budget._initialize_token_budget()

    def test_sliding_region_budget_adds_region_before_stopping(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 180
        summarizer._token_len_cache = {}
        summarizer._cacheable_token_len = lambda cacheable: 80
        summarizer._cacheable_token_lens = lambda cacheables: [80 for _ in cacheables]
        source_cacheables = [
            CacheableChunk(id="s0", text="alpha"),
            CacheableChunk(id="s1", text="beta"),
            CacheableChunk(id="s2", text="gamma"),
        ]
        first_region = {
            "chunk_idx": 0,
            "region_id": "chunk0::sliding_region_0",
            "parent_doc_id": "chunk0",
            "selected_indices": (0, 1),
            "source_cacheables": source_cacheables,
        }
        second_region = {
            "chunk_idx": 0,
            "region_id": "chunk0::sliding_region_2",
            "parent_doc_id": "chunk0",
            "selected_indices": (2,),
            "source_cacheables": source_cacheables,
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, first_region), (9.0, second_region)],
            final_token_budget=100,
        )

        self.assertEqual(len(selected), 1)
        chunk_idx, cacheable = selected[0]
        self.assertEqual(chunk_idx, 0)
        self.assertEqual(cacheable.sentence_ids, ["s0", "s1"])
        self.assertEqual(cacheable.text, "alpha beta")

        selected, scores = summarizer._select_sliding_regions_with_scores(
            [(10.0, first_region), (9.0, second_region)],
            final_token_budget=100,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(scores, {0: [10.0]})

    def test_region_group_order_supports_max_and_sum(self):
        docs = [
            RetrievableChunk(id="chunk0", text="zero"),
            RetrievableChunk(id="chunk1", text="one"),
            RetrievableChunk(id="chunk2", text="two"),
        ]
        scores = {0: [6.0, 6.0], 1: [10.0], 2: [5.0]}
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)

        summarizer.region_group_order = "retrieval"
        self.assertEqual(summarizer._order_document_groups(docs, scores), docs)

        summarizer.region_group_order = "max"
        max_order = summarizer._order_document_groups(docs, scores)
        self.assertEqual([doc.id for doc in max_order], ["chunk1", "chunk0", "chunk2"])

        summarizer.region_group_order = "sum"
        sum_order = summarizer._order_document_groups(docs, scores)
        self.assertEqual([doc.id for doc in sum_order], ["chunk0", "chunk1", "chunk2"])

    def test_rerank_and_region_reuses_region_flow_with_chunk_gate(self):
        self.assertIs(
            ColBERTRerankAndRegionCompressor.compress_batch_top_k_docs,
            ColBERTSlidingRegionCompressor.compress_batch_top_k_docs,
        )
        docs = [
            RetrievableChunk(id="chunk0", text="zero"),
            RetrievableChunk(id="chunk1", text="one"),
            RetrievableChunk(id="chunk2", text="two"),
        ]

        sliding = object.__new__(ColBERTSlidingRegionCompressor)
        sliding_profile = {}
        self.assertEqual(
            sliding._region_chunk_indices(
                docs, query_vector=None, profile=sliding_profile
            ),
            {0, 1, 2},
        )
        self.assertEqual(sliding_profile, {})

        combined = object.__new__(ColBERTRerankAndRegionCompressor)
        combined.rerank_keep = 2
        combined._rerank_chunk_indices = lambda docs, query_vector, profile: [
            2,
            0,
            1,
        ]
        combined_profile = {}
        self.assertEqual(
            combined._region_chunk_indices(
                docs, query_vector=None, profile=combined_profile
            ),
            {0, 2},
        )
        self.assertEqual(combined_profile["rerank_kept_chunk_count"], 2)

    def test_rerank_and_region_gates_budget_and_region_construction(self):
        docs = [
            RetrievableChunk(id="chunk0", text="zero"),
            RetrievableChunk(id="chunk1", text="one"),
            RetrievableChunk(id="chunk2", text="two"),
        ]

        class FakeEncoder:
            def encode_queries(self, queries):
                return [None for _ in queries]

        combined = object.__new__(ColBERTRerankAndRegionCompressor)
        combined.encoder = FakeEncoder()
        combined.rerank_keep = 1
        combined._rerank_chunk_indices = lambda docs, query_vector, profile: [
            2,
            0,
            1,
        ]
        budget_docs = []
        combined._resolve_final_token_budget = lambda docs: (
            budget_docs.append([doc.id for doc in docs]) or 100
        )
        region_docs = []
        combined._sliding_regions_for_doc = lambda doc, chunk_idx, profile: (
            region_docs.append((chunk_idx, doc.id)) or []
        )

        output = combined.compress_batch_top_k_docs([docs], ["query"])

        self.assertEqual(budget_docs, [["chunk2"]])
        self.assertEqual(region_docs, [(2, "chunk2")])
        self.assertEqual(combined.last_profile["rerank_kept_chunk_count"], 1)
        self.assertEqual(len(output[0]), 3)
        self.assertTrue(all(not doc.cacheables for doc in output[0]))

    def test_budget_token_lengths_are_cached_by_cacheable_id(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        tokenizer = CountingBudgetTokenizer()
        summarizer.budget_tokenizer = tokenizer
        summarizer._token_len_cache = {}
        cacheables = [
            CacheableChunk(id="s0", text="alpha beta"),
            CacheableChunk(id="s1", text="gamma"),
        ]

        first = summarizer._cacheable_token_lens(cacheables)
        second = summarizer._cacheable_token_lens(cacheables)

        self.assertEqual(first, [2, 1])
        self.assertEqual(second, [2, 1])
        self.assertEqual(len(tokenizer.calls), 1)

    def test_budget_token_lengths_use_offline_prompt_token_count(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        tokenizer = CountingBudgetTokenizer()
        summarizer.budget_tokenizer = tokenizer
        summarizer.budget_tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        summarizer._token_len_cache = {}
        cacheables = [
            CacheableChunk(
                id="s0",
                text="alpha beta",
                prompt_token_count=9,
                prompt_tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            ),
            CacheableChunk(
                id="s1",
                text="gamma",
                prompt_token_count=4,
                prompt_tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            ),
        ]

        lengths = summarizer._cacheable_token_lens(cacheables)

        self.assertEqual(lengths, [9, 4])
        self.assertEqual(tokenizer.calls, [])

    def test_budget_token_lengths_ignore_stored_count_for_wrong_tokenizer(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        tokenizer = CountingBudgetTokenizer()
        summarizer.budget_tokenizer = tokenizer
        summarizer.budget_tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
        summarizer._token_len_cache = {}
        cacheables = [
            CacheableChunk(
                id="s0",
                text="alpha beta",
                prompt_token_count=99,
                prompt_tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            )
        ]

        lengths = summarizer._cacheable_token_lens(cacheables)

        self.assertEqual(lengths, [2])
        self.assertEqual(tokenizer.calls, [["alpha beta\n\n"]])

    def test_encoder_window_budget_overflow_stops_expansion(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        encoder.doc_maxlen = 40
        encoder.doc_token_overhead = 0
        encoder.token_counts_without_specials = lambda sentences: [
            int(sentence) for sentence in sentences
        ]

        specs = encoder.build_centered_windows(
            ["10", "1000", "10", "10", "10"],
            window_token_budget=40,
        )

        self.assertEqual(specs[2].selected_indices, [2])
        self.assertEqual(specs[2].addition_order, [2])

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
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 40
        summarizer._sliding_region_spec_cache = {}
        summarizer.artifact = FakeRegionArtifact([(0, (0, 1)), (2, (2,))])
        summarizer.encoder = FailingWindowEncoder()

        specs = summarizer._cached_sliding_region_specs(doc, doc.cacheables)

        self.assertEqual(specs, [(0, (0, 1)), (2, (2,))])
        self.assertEqual(summarizer.artifact.calls, [("chunk0", 40)])

    def test_sliding_region_spec_cache_key_includes_cacheable_ids(self):
        class CacheAwareArtifact:
            def region_specs_for_doc(self, doc, token_budget):
                del token_budget
                first_id = doc.cacheables[0].id
                if first_id.endswith("sent_0"):
                    return [(0, (0, 1))]
                return [(0, (0,)), (1, (1,))]

        first_doc = RetrievableChunk(
            id="chunk0",
            text="10 10",
            cacheables=[
                CacheableChunk(id="chunk0::sent_0", text="10"),
                CacheableChunk(id="chunk0::sent_1", text="10"),
            ],
        )
        second_doc = RetrievableChunk(
            id="chunk0",
            text="1000 delta",
            cacheables=[
                CacheableChunk(id="chunk0::sent_2", text="1000"),
                CacheableChunk(id="chunk0::sent_3", text="10"),
            ],
        )
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 40
        summarizer._sliding_region_spec_cache = {}
        summarizer.artifact = CacheAwareArtifact()

        first_specs = summarizer._cached_sliding_region_specs(
            first_doc, first_doc.cacheables
        )
        second_specs = summarizer._cached_sliding_region_specs(
            second_doc, second_doc.cacheables
        )

        self.assertEqual(first_specs, [(0, (0, 1))])
        self.assertEqual(second_specs, [(0, (0,)), (1, (1,))])
        self.assertEqual(len(summarizer._sliding_region_spec_cache), 2)

    def test_window_bounded_region_specs_drop_boundary_strict_subsets(self):
        cacheable_ids = [f"doc::sent_{idx}" for idx in range(2, 6)]
        window_ids_by_cacheable_id = {
            "doc::sent_2": [
                "doc::sent_0",
                "doc::sent_1",
                "doc::sent_2",
                "doc::sent_3",
                "doc::sent_4",
            ],
            "doc::sent_3": [
                "doc::sent_1",
                "doc::sent_2",
                "doc::sent_3",
                "doc::sent_4",
                "doc::sent_5",
            ],
            "doc::sent_4": [
                "doc::sent_2",
                "doc::sent_3",
                "doc::sent_4",
                "doc::sent_5",
                "doc::sent_6",
            ],
            "doc::sent_5": [
                "doc::sent_3",
                "doc::sent_4",
                "doc::sent_5",
                "doc::sent_6",
                "doc::sent_7",
            ],
        }

        bounded = _window_bounded_region_index_specs(
            cacheable_ids=cacheable_ids,
            window_ids_by_cacheable_id=window_ids_by_cacheable_id,
        )

        self.assertEqual([selected for _, selected in bounded], [(0, 1, 2, 3)])

    def test_window_bounded_region_specs_clip_tail_chunk_to_chunk_extent(self):
        cacheable_ids = [f"doc::sent_{idx}" for idx in range(5, 8)]
        window_ids_by_cacheable_id = {
            "doc::sent_5": [
                "doc::sent_3",
                "doc::sent_4",
                "doc::sent_5",
                "doc::sent_6",
                "doc::sent_7",
            ],
            "doc::sent_6": ["doc::sent_4", "doc::sent_5", "doc::sent_6", "doc::sent_7"],
            "doc::sent_7": ["doc::sent_5", "doc::sent_6", "doc::sent_7"],
        }

        bounded = _window_bounded_region_index_specs(
            cacheable_ids=cacheable_ids,
            window_ids_by_cacheable_id=window_ids_by_cacheable_id,
        )

        self.assertEqual([selected for _, selected in bounded], [(0, 1, 2)])

    def test_add_region_specs_uses_stored_window_ids_without_doc_pt_payloads(self):
        with TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "colbert_window"
            data_dir = artifact_dir / "data"
            data_dir.mkdir(parents=True)
            (artifact_dir / "index.json").write_text(
                json.dumps(
                    {
                        "format": ARTIFACT_FORMAT,
                        "model_name": "colbert-ir/colbertv2.0",
                        "checkpoint_name": "colbert-ir/colbertv2.0",
                        "repo_path": "",
                        "window_token_budget": 180,
                    }
                ),
                encoding="utf-8",
            )
            (data_dir / "index.json").write_text(
                json.dumps(
                    {
                        "format": "colbert_window_data_build_v1",
                        "num_cacheables": 3,
                        "num_tokens": 3,
                        "embedding_dim": 1,
                        "vectors_file": "vectors.fp16.bin",
                        "offsets_file": "offsets.npy",
                        "metadata_file": "metadata.sqlite3",
                        "region_token_budget": None,
                        "region_spec_chunk_count": 0,
                        "region_spec_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            metadata_writer = ColBERTMetadataWriter(data_dir / "metadata.sqlite3")
            metadata_writer.add_cacheables(
                [
                    ("s0", 0, ["s0", "s1"]),
                    ("s1", 1, ["s0", "s1", "s2"]),
                    ("s2", 2, ["s1", "s2"]),
                ]
            )
            metadata_writer.close()
            db_groups = [
                (
                    "chunk0",
                    [
                        CacheableChunk(id="s0", text="a"),
                        CacheableChunk(id="s1", text="b"),
                        CacheableChunk(id="s2", text="c"),
                    ],
                )
            ]

            with (
                patch.object(
                    colbert_materializer,
                    "iter_db_cacheable_groups",
                    return_value=iter(db_groups),
                ),
                patch.object(
                    colbert_materializer.torch,
                    "load",
                    side_effect=AssertionError("data path must not load .pt"),
                ),
            ):
                summary = add_region_specs_to_colbert_window_data(
                    data_dir=data_dir,
                    db_dir=Path(tmpdir) / "db",
                    region_token_budget=180,
                )

            self.assertEqual(summary["region_spec_reuse_mode"], "document_window_bound")
            region_payloads = json.loads(
                (data_dir / summary["region_payloads_file"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                region_payloads["chunk0"]["specs"],
                [[1, [0, 1, 2]]],
            )


if __name__ == "__main__":
    unittest.main()
