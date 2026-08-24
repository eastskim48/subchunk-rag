import json
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk, RetrievableChunk
from colbert_artifact import (
    ARTIFACT_FORMAT,
    DATA_ARTIFACT_FORMAT,
    ColBERTWindowArtifact,
)
from compressor.methods.colbert import (
    ColBERTRerankAndRegionCompressor,
    ColBERTRerankCompressor,
    ColBERTSubchunkCompressor,
    ColBERTSlidingRegionCompressor,
    _resolve_configured_retrieval_chunk_size,
)
from compressor.methods.colbert.scoring import score_maxsim, sentence_token_maxsim
from compressor.methods.dense import DenseCompressor
from compressor.methods.summarization import Summarizer
from colbert_artifact import build_db_manifest_reference
from colbert_metadata import (
    SQLITE_LOOKUP_BATCH_SIZE,
    ColBERTMetadataReader,
    ColBERTMetadataWriter,
    write_split_metadata_from_sqlite,
)
from encoder.colbert import ColBERTEncoder
from materialize.db_manifest import build_db_manifest, write_db_build_manifest
from prompt import PromptProcessor
import compressor.factory as compressor_factory


class FakeBudgetTokenizer:
    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        verbose=False,
    ):
        self.last_call = {
            "padding": padding,
            "truncation": truncation,
            "add_special_tokens": add_special_tokens,
            "verbose": verbose,
        }
        if isinstance(texts, str):
            return {"input_ids": [0] * len(texts.split())}
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class FakeQueryEncoder:
    doc_token_overhead = 0
    max_length = 180
    init_kwargs = []

    def __init__(self, *args, **kwargs):
        del args
        self.init_kwargs.append(kwargs)

    def encode_queries(self, queries):
        return [torch.tensor([[1.0, 0.0]], dtype=torch.float32) for _ in queries]


class FakeSidecarCountingArtifact:
    def __init__(self):
        self.index = {
            "checkpoint_name": "colbert-ir/colbertv2.0",
            "source_tokenizer_name": "artifact-tokenizer",
        }
        self.retrievable_vectors_cache = {}


class ColBERTWindowOptimizationTest(unittest.TestCase):
    def test_sqlite_metadata_lookups_exceed_single_statement_limit(self):
        record_count = SQLITE_LOOKUP_BATCH_SIZE + 7
        records = [
            (f"id-{index}", index, [f"window-{index}"]) for index in range(record_count)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.sqlite3"
            writer = ColBERTMetadataWriter(metadata_path)
            writer.add_cacheables(records)
            writer.close()
            reader = ColBERTMetadataReader(metadata_path)
            cacheable_ids = [record[0] for record in records]

            rows = reader.rows_for_cacheable_ids(cacheable_ids)
            windows = reader.window_ids_for_cacheable_ids(cacheable_ids)
            reader.close()

        self.assertEqual(rows, list(range(record_count)))
        self.assertEqual(
            windows,
            [[f"window-{index}"] for index in range(record_count)],
        )

    def test_adaptive_query_encoding_uses_q32_floor_and_configured_cap(self):
        class FakeRawTokenizer:
            def __call__(self, queries, **kwargs):
                del kwargs
                lengths = {"short": 10, "medium": 35, "long": 200}
                return {"input_ids": [list(range(lengths[query])) for query in queries]}

        class FakeCheckpoint:
            query_tokenizer = type(
                "FakeQueryTokenizer", (), {"tok": FakeRawTokenizer()}
            )()

            def queryFromText(self, queries, bsize, to_cpu):
                del bsize, to_cpu
                return torch.zeros((len(queries), 128, 2), dtype=torch.float32)

        encoder = object.__new__(ColBERTEncoder)
        encoder.checkpoint = FakeCheckpoint()
        encoder.batch_size = 3
        encoder.query_minlen = 32
        encoder.query_maxlen = 128

        vectors = encoder.encode_queries(["short", "medium", "long"])

        self.assertEqual([len(vector) for vector in vectors], [32, 38, 128])

    def test_fixed_query_encoding_keeps_configured_representation_length(self):
        class FakeCheckpoint:
            inference_mode_enabled = False

            def queryFromText(self, queries, bsize, to_cpu):
                del bsize, to_cpu
                self.inference_mode_enabled = torch.is_inference_mode_enabled()
                return torch.zeros((len(queries), 100, 2), dtype=torch.float32)

        encoder = object.__new__(ColBERTEncoder)
        encoder.checkpoint = FakeCheckpoint()
        encoder.batch_size = 2
        encoder.query_minlen = None
        encoder.query_maxlen = 100

        vectors = encoder.encode_queries(["first", "second"])

        self.assertEqual([len(vector) for vector in vectors], [100, 100])
        self.assertTrue(encoder.checkpoint.inference_mode_enabled)

    def test_cpu_checkpoint_disables_official_cuda_amp_context(self):
        amp_manager = type("FakeAMPManager", (), {"activated": True})()
        checkpoint = type("FakeCheckpoint", (), {"amp_manager": amp_manager})()

        ColBERTEncoder._configure_checkpoint_inference(
            checkpoint=checkpoint,
            requested_device=torch.device("cpu"),
        )

        self.assertFalse(amp_manager.activated)

    def test_factory_registers_dense_without_removed_compare_names(self):
        self.assertIs(compressor_factory.COMPRESSOR_TYPES["dense"], DenseCompressor)
        self.assertIs(compressor_factory.COMPRESSOR_TYPES["summ"], Summarizer)
        self.assertNotIn("front", compressor_factory.COMPRESSOR_TYPES)

    def test_sliding_region_preserves_source_parent_document_id(self):
        doc = RetrievableChunk(
            id="doc0.txt::ret_3",
            text="evidence",
            cacheables=[
                CacheableChunk(
                    id="doc0.txt::sent_1",
                    text="evidence",
                    parent_doc_id="doc0.txt",
                )
            ],
            metadata={"parent_doc_id": "doc0.txt"},
        )

        parent_doc_id = ColBERTSlidingRegionCompressor._source_parent_document_id(
            doc, doc.cacheables
        )

        self.assertEqual(parent_doc_id, "doc0.txt")
        self.assertNotIn("compare", compressor_factory.COMPRESSOR_TYPES)
        self.assertNotIn("compare_all", compressor_factory.COMPRESSOR_TYPES)
        self.assertNotIn(
            "compare_all_materialized", compressor_factory.COMPRESSOR_TYPES
        )

    def test_factory_registers_colbert_subchunk_without_old_window_alias(self):
        self.assertIs(
            compressor_factory.COMPRESSOR_TYPES["colbert_subchunk"],
            ColBERTSubchunkCompressor,
        )
        self.assertNotIn("colbert_window", compressor_factory.COMPRESSOR_TYPES)
        self.assertNotIn("colbert_window_budget", compressor_factory.COMPRESSOR_TYPES)

    def test_colbert_reranker_scores_all_subchunk_vectors_per_doc(self):
        class FakeArtifact:
            def __init__(self, vectors_by_doc):
                self.vectors_by_doc = vectors_by_doc

            def vectors_for_doc(self, doc):
                return self.vectors_by_doc[str(doc.id)]

        summarizer = object.__new__(ColBERTRerankCompressor)
        summarizer.keep = 2
        summarizer.encoder = FakeQueryEncoder()
        summarizer.artifact = FakeArtifact(
            {
                "doc-a": [
                    torch.tensor([[10.0, 0.0]], dtype=torch.float16),
                    torch.tensor([[0.0, 10.0]], dtype=torch.float16),
                ],
                "doc-b": [
                    torch.tensor([[8.0, 8.0]], dtype=torch.float16),
                ],
                "doc-c": [
                    torch.tensor([[1.0, 1.0]], dtype=torch.float16),
                ],
            }
        )
        docs = [
            RetrievableChunk(id="doc-c", text="c"),
            RetrievableChunk(id="doc-b", text="b"),
            RetrievableChunk(id="doc-a", text="a"),
        ]

        with patch(
            "compressor.methods.colbert.rerank.sentence_token_maxsim",
            wraps=sentence_token_maxsim,
        ) as sentence_scorer:
            output = summarizer.compress_batch_top_k_docs([docs], ["query"])[0]

        self.assertEqual(sentence_scorer.call_count, 1)
        self.assertEqual([str(doc.id) for doc in output], ["doc-a", "doc-b"])
        self.assertTrue(
            all(
                selected is not source
                for selected, source in zip(output, [docs[2], docs[1]])
            )
        )
        self.assertEqual(summarizer.last_profile["selected_doc_count"], 2)

    def test_rerank_and_region_reuses_coarse_sentence_scores_for_regions(self):
        class FakeArtifact:
            def __init__(self, vectors_by_doc):
                self.vectors_by_doc = vectors_by_doc

            def vectors_for_doc(self, doc):
                return self.vectors_by_doc[str(doc.id)]

        summarizer = object.__new__(ColBERTRerankAndRegionCompressor)
        first_vectors = [
            torch.tensor([[5.0, 0.0]], dtype=torch.float16),
            torch.tensor([[0.0, 5.0]], dtype=torch.float16),
        ]
        second_vectors = [torch.tensor([[4.0, 4.0]], dtype=torch.float16)]
        summarizer.artifact = FakeArtifact(
            {"doc-a": first_vectors, "doc-b": second_vectors}
        )
        first_cacheables = [
            CacheableChunk(id="doc-a::sent_0", text="alpha"),
            CacheableChunk(id="doc-a::sent_1", text="beta"),
        ]
        docs = [
            RetrievableChunk(
                id="doc-a", text="alpha beta", cacheables=first_cacheables
            ),
            RetrievableChunk(
                id="doc-b",
                text="gamma",
                cacheables=[CacheableChunk(id="doc-b::sent_0", text="gamma")],
            ),
        ]
        query_vector = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        ranked_indices = summarizer._rerank_chunk_indices(docs, query_vector, {})
        regions = [
            {
                "chunk_idx": 0,
                "selected_indices": (0, 1),
                "source_cacheables": first_cacheables,
                "source_vectors": first_vectors,
            }
        ]
        with patch(
            "compressor.methods.colbert.region.sentence_token_maxsim"
        ) as region_sentence_scorer:
            region_scores = summarizer._score_sliding_regions_vectorized(
                query_vector, regions
            )

        self.assertEqual(ranked_indices, [0, 1])
        region_sentence_scorer.assert_not_called()
        self.assertEqual(
            region_scores[0], score_maxsim(query_vector, torch.cat(first_vectors))
        )

    def test_factory_registers_only_the_new_colbert_rerank_names(self):
        self.assertIs(
            compressor_factory.COMPRESSOR_TYPES["colbert_rerank"],
            ColBERTRerankCompressor,
        )
        self.assertIs(
            compressor_factory.COMPRESSOR_TYPES["rerank_and_region"],
            ColBERTRerankAndRegionCompressor,
        )
        self.assertNotIn("rerank_and_reion", compressor_factory.COMPRESSOR_TYPES)
        self.assertNotIn(
            "colbert_window_chunk_rerank", compressor_factory.COMPRESSOR_TYPES
        )
        self.assertNotIn("rerank_pre_filter", compressor_factory.COMPRESSOR_TYPES)
        self.assertNotIn(
            "colbert_sliding_region_rerank_pre_filter",
            compressor_factory.COMPRESSOR_TYPES,
        )
        self.assertNotIn(
            "colbert_sliding_region_rerank_post_filter",
            compressor_factory.COMPRESSOR_TYPES,
        )

    def test_colbert_rerank_paths_require_positive_shared_keep(self):
        with patch.dict(os.environ, {"COLBERT_RERANK_KEEP": "0"}):
            for summarizer_type in (
                ColBERTRerankCompressor,
                ColBERTRerankAndRegionCompressor,
            ):
                with self.subTest(summarizer_type=summarizer_type.__name__):
                    with self.assertRaisesRegex(
                        ValueError, "COLBERT_RERANK_KEEP must be positive"
                    ):
                        summarizer_type()

    def test_factory_always_warms_once_and_clears_before_each_batch(self):
        class FakeCompressor:
            def __init__(self):
                self.warmup_count = 0
                self.clear_count = 0

            def warmup_query_encoder(self):
                self.warmup_count += 1
                return 0.0

            def clear_inter_batch_cache(self):
                self.clear_count += 1

            def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
                del batch_queries
                return batch_top_k_docs

        with (
            patch.object(
                compressor_factory, "COMPRESSOR_TYPES", {"fake": FakeCompressor}
            ),
            patch.object(compressor_factory, "compressor", None),
            patch.object(compressor_factory, "compressor_warmed", False),
        ):
            active = compressor_factory.initialize_compressor("fake")
            compressor_factory.initialize_compressor("fake")
            compressor_factory.compress_docs(["query"], [[]], "fake")
            compressor_factory.compress_docs(["query"], [[]], "fake")

            self.assertEqual(active.warmup_count, 1)
            self.assertEqual(active.clear_count, 2)

    def test_legacy_window_artifact_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "index.json").write_text(
                json.dumps({"format": "unsupported_colbert_window_v0"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported ColBERT"):
                ColBERTWindowArtifact(artifact_dir)

    def test_artifact_requires_data_store_and_matching_db_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            data_dir = artifact_dir / "data"
            data_dir.mkdir()
            db_dir = artifact_dir / "db"
            write_db_build_manifest(
                db_dir,
                build_db_manifest(
                    splitter="sentence",
                    merger=None,
                    cacheable_chunk_size=None,
                    retrievable_chunk_size=512,
                    max_subchunk_tokens=180,
                    tokenizer_name="tokenizer",
                    dummy_bos_count=4,
                    sentence_cache_token_format="legacy",
                    deduplicate_documents_by_hash=True,
                    embedding_backend="default",
                    db_batch_size=256,
                    embedding_device="cpu",
                    embedding_batch_size=32,
                ),
            )
            _, db_manifest_reference = build_db_manifest_reference(
                db_dir=db_dir, artifact_dir=artifact_dir
            )
            np.zeros((5, 4), dtype=np.float16).tofile(data_dir / "vectors.fp16.bin")
            np.save(data_dir / "offsets.npy", np.asarray([0, 2, 5], dtype=np.int64))
            metadata_writer = ColBERTMetadataWriter(data_dir / "metadata.sqlite3")
            metadata_writer.add_cacheables(
                [
                    ("doc0.txt::sent_0", 0, ["doc0.txt::sent_0"]),
                    (
                        "doc0.txt::sent_1",
                        1,
                        ["doc0.txt::sent_0", "doc0.txt::sent_1"],
                    ),
                ]
            )
            metadata_writer.replace_regions(
                [
                    (
                        "doc0.txt::ret_0",
                        ["doc0.txt::sent_0", "doc0.txt::sent_1"],
                        [(0, (0,)), (1, (0, 1))],
                    )
                ]
            )
            metadata_writer.close()
            metadata_reader = ColBERTMetadataReader(data_dir / "metadata.sqlite3")
            split_fields = write_split_metadata_from_sqlite(metadata_reader, data_dir)
            metadata_reader.close()
            (data_dir / "index.json").write_text(
                json.dumps(
                    {
                        "format": DATA_ARTIFACT_FORMAT,
                        "embedding_dim": 4,
                        "num_tokens": 5,
                        "num_cacheables": 2,
                        "vectors_file": "vectors.fp16.bin",
                        "offsets_file": "offsets.npy",
                        **split_fields,
                        "region_token_budget": 180,
                        "region_spec_chunk_count": 1,
                        "region_spec_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "index.json").write_text(
                json.dumps(
                    {
                        "format": ARTIFACT_FORMAT,
                        "db_manifest": db_manifest_reference,
                        "embedding_dim": 4,
                        "data_dir": "data",
                        "window_token_budget": 180,
                        "official_query_maxlen": 32,
                    }
                ),
                encoding="utf-8",
            )
            artifact = ColBERTWindowArtifact(artifact_dir)
            artifact.validate_db_manifest(db_dir)
            other_db_dir = artifact_dir / "other-db"
            write_db_build_manifest(other_db_dir, artifact.db_manifest)
            with self.assertRaisesRegex(ValueError, "manifest path"):
                artifact.validate_db_manifest(other_db_dir)

            doc = RetrievableChunk(
                id="doc0.txt::ret_0",
                text="",
                cacheables=[
                    CacheableChunk(id="doc0.txt::sent_0", text="a"),
                    CacheableChunk(id="doc0.txt::sent_1", text="b"),
                ],
            )
            first_vectors = artifact.vectors_for_doc(doc)
            second_vectors = artifact.vectors_for_doc(doc)
            direct_vectors = artifact.data.vectors_for_cacheable_ids(
                ["doc0.txt::sent_0", "doc0.txt::sent_1"]
            )
            self.assertEqual(
                artifact.region_specs_for_doc(doc, 180),
                [(0, (0,)), (1, (0, 1))],
            )
            with self.assertRaisesRegex(ValueError, "region budget"):
                artifact.region_specs_for_doc(doc, 120)
            missing_doc = RetrievableChunk(
                id="missing::ret_0",
                text="",
                cacheables=[CacheableChunk(id="missing::sent_0", text="x")],
            )
            with self.assertRaisesRegex(ValueError, "missing runtime chunk"):
                artifact.region_specs_for_doc(missing_doc, 180)
            mismatched_doc = RetrievableChunk(
                id="doc0.txt::ret_0",
                text="",
                cacheables=[CacheableChunk(id="doc0.txt::sent_0", text="a")],
            )
            with self.assertRaisesRegex(ValueError, "cacheable IDs"):
                artifact.region_specs_for_doc(mismatched_doc, 180)

            data_index_path = data_dir / "index.json"
            data_index = json.loads(data_index_path.read_text(encoding="utf-8"))
            data_index["region_token_budget"] = 120
            data_index_path.write_text(json.dumps(data_index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match artifact"):
                ColBERTWindowArtifact(artifact_dir)

        self.assertIs(first_vectors, second_vectors)
        self.assertEqual([len(vector) for vector in direct_vectors], [2, 3])

    def test_vectorized_region_score_matches_concatenated_maxsim(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
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
            self.assertEqual(score, expected_score)

    def test_exhaustive_small_region_scores_match_concatenated_maxsim(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
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
                for score, expected_score in zip(vectorized_scores, expected_scores):
                    if expected_score == float("-inf"):
                        self.assertEqual(score, float("-inf"))
                    else:
                        self.assertEqual(score, expected_score)

    def test_vectorized_scores_do_not_collide_on_duplicate_cacheable_ids(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
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
        expected_scores = [
            score_maxsim(query_vector, first_vectors[0]),
            score_maxsim(query_vector, second_vectors[0]),
        ]

        for score, expected_score in zip(vectorized_scores, expected_scores):
            self.assertAlmostEqual(score, expected_score, places=5)

    def test_colbert_region_document_restores_source_order_after_score_order_selection(
        self,
    ):
        source_cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha"),
            CacheableChunk(id="doc0::sent_1", text="beta"),
            CacheableChunk(id="doc0::sent_2", text="gamma"),
        ]
        doc = RetrievableChunk(
            id="doc0", text="alpha beta gamma", cacheables=source_cacheables
        )
        selected_cacheables = [
            CacheableChunk(
                id="doc0::region_2",
                text="gamma",
                sentence_ids=["doc0::sent_2"],
            ),
            CacheableChunk(
                id="doc0::region_0",
                text="alpha",
                sentence_ids=["doc0::sent_0"],
            ),
            CacheableChunk(
                id="doc0::region_1",
                text="beta",
                sentence_ids=["doc0::sent_1"],
            ),
        ]

        output_doc = ColBERTSlidingRegionCompressor._build_region_document(
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
        self.assertTrue(
            all(
                cacheable.chunk_start is None and cacheable.chunk_end is None
                for cacheable in output_doc.cacheables
            )
        )

    def test_colbert_selection_splits_noncontiguous_novel_region_runs(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 40
        summarizer._cacheable_token_lens = lambda cacheables: [1 for _ in cacheables]
        summarizer._cacheable_token_len = lambda cacheable: 1
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
            final_token_budget=3,
        )
        output_doc = ColBERTSlidingRegionCompressor._build_region_document(
            RetrievableChunk(
                id="doc0", text="alpha beta gamma", cacheables=source_cacheables
            ),
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
        self.assertTrue(
            all(cacheable.chunk_end is None for cacheable in output_doc.cacheables)
        )

    def test_colbert_budget_selection_keeps_selected_region_when_it_exceeds_budget(
        self,
    ):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
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
        output_doc = ColBERTSlidingRegionCompressor._build_region_document(
            RetrievableChunk(
                id="doc0", text="alpha beta gamma", cacheables=source_cacheables
            ),
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
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
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
        output_doc = ColBERTSlidingRegionCompressor._build_region_document(
            RetrievableChunk(
                id="doc0", text="alpha beta gamma", cacheables=source_cacheables
            ),
            [cacheable for _, cacheable in large],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha", "beta", "gamma"],
        )

    def test_colbert_budget_selection_exhaustive_small_region_invariants(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 40
        summarizer._cacheable_token_lens = lambda cacheables: [
            cacheable.chunk_size for cacheable in cacheables
        ]
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        source_cacheables = [
            CacheableChunk(
                id=f"doc0::sent_{idx}",
                text=f"s{idx}",
                chunk_size=token_len,
                chunk_start=idx * 10,
                chunk_end=idx * 10 + 5,
            )
            for idx, token_len in enumerate([1, 2, 1, 3])
        ]
        regions = []
        for center_idx, selected_indices in enumerate(
            [(0,), (0, 1), (1, 2), (1, 2, 3), (3,)]
        ):
            regions.append(
                {
                    "chunk_idx": 0,
                    "center_idx": center_idx,
                    "region_id": f"doc0::region_{center_idx}",
                    "parent_doc_id": "doc0",
                    "selected_indices": selected_indices,
                    "source_cacheables": source_cacheables,
                }
            )
        scored_regions = [
            (score, region) for score, region in zip([9, 8, 7, 6, 5], regions)
        ]

        for final_budget in range(1, 8):
            with self.subTest(final_budget=final_budget):
                selected = summarizer._select_sliding_regions(
                    scored_regions,
                    final_token_budget=final_budget,
                )
                selected_cacheables = [cacheable for _, cacheable in selected]
                selected_sentence_ids = [
                    sentence_id
                    for cacheable in selected_cacheables
                    for sentence_id in cacheable.sentence_ids
                ]
                self.assertEqual(
                    len(selected_sentence_ids), len(set(selected_sentence_ids))
                )
                for cacheable in selected_cacheables:
                    indices = [
                        int(sentence_id.rsplit("_", 1)[1])
                        for sentence_id in cacheable.sentence_ids
                    ]
                    self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))
                    self.assertEqual(
                        cacheable.text, " ".join(f"s{idx}" for idx in indices)
                    )

                output_doc = ColBERTSlidingRegionCompressor._build_region_document(
                    RetrievableChunk(
                        id="doc0",
                        text=" ".join(f"s{i}" for i in range(4)),
                        cacheables=source_cacheables,
                    ),
                    selected_cacheables,
                )
                ordered_ids = [
                    sentence_id
                    for cacheable in output_doc.cacheables
                    for sentence_id in cacheable.sentence_ids
                ]
                selected_id_set = set(ordered_ids)
                self.assertEqual(
                    ordered_ids,
                    [
                        source.id
                        for source in source_cacheables
                        if source.id in selected_id_set
                    ],
                )
                self.assertTrue(
                    all(
                        cacheable.chunk_start is None and cacheable.chunk_end is None
                        for cacheable in output_doc.cacheables
                    )
                )

                used_tokens = sum(
                    source_cacheables[int(sentence_id.rsplit("_", 1)[1])].chunk_size
                    for sentence_id in selected_sentence_ids
                )
                all_tokens = sum(
                    cacheable.chunk_size for cacheable in source_cacheables
                )
                self.assertTrue(
                    used_tokens >= final_budget or used_tokens == all_tokens,
                    (final_budget, used_tokens, all_tokens, selected_sentence_ids),
                )

    def test_colbert_budget_token_len_uses_prompt_visible_text_not_sidecar(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        cacheable = CacheableChunk(id="doc0::sent_0", text="one two three")

        self.assertEqual(summarizer._cacheable_token_len(cacheable), 3)
        self.assertEqual(
            summarizer.budget_tokenizer.last_call,
            {
                "padding": False,
                "truncation": False,
                "add_special_tokens": False,
                "verbose": False,
            },
        )

    def test_colbert_budget_selection_keeps_single_cacheable_over_final_budget(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 180
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        long_text = " ".join(["long"] * 101)
        source_cacheables = [
            CacheableChunk(
                id="doc0::sent_0",
                text=long_text,
                chunk_start=0,
                chunk_end=len(long_text),
            ),
            CacheableChunk(
                id="doc0::sent_1",
                text="short evidence",
                chunk_start=len(long_text) + 1,
                chunk_end=len(long_text) + 15,
            ),
        ]
        long_region = {
            "chunk_idx": 0,
            "center_idx": 0,
            "region_id": "doc0::region_0",
            "parent_doc_id": "doc0",
            "selected_indices": (0,),
            "source_cacheables": source_cacheables,
        }
        short_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_1",
            "parent_doc_id": "doc0",
            "selected_indices": (1,),
            "source_cacheables": source_cacheables,
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, long_region), (9.0, short_region)],
            final_token_budget=100,
        )

        self.assertEqual([cacheable.text for _, cacheable in selected], [long_text])

    def test_colbert_sliding_region_compress_respects_absolute_prompt_budget(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.final_token_budget = 3
        summarizer.retain_token_ratio = None
        summarizer.region_token_budget = 180
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        summarizer.encoder = FakeQueryEncoder()
        cacheables = [
            CacheableChunk(id="doc0::sent_0", text="one", chunk_start=0, chunk_end=3),
            CacheableChunk(id="doc0::sent_1", text="two", chunk_start=4, chunk_end=7),
            CacheableChunk(
                id="doc0::sent_2", text="three", chunk_start=8, chunk_end=13
            ),
            CacheableChunk(
                id="doc0::sent_3", text="four", chunk_start=14, chunk_end=18
            ),
        ]
        doc = RetrievableChunk(
            id="doc0::ret_0", text="one two three four", cacheables=cacheables
        )

        def fake_regions(doc, chunk_idx, profile=None):
            del doc, profile
            return [
                {
                    "chunk_idx": chunk_idx,
                    "center_idx": 0,
                    "region_id": "doc0::region_0_3",
                    "parent_doc_id": "doc0::ret_0",
                    "selected_indices": (0, 1, 2, 3),
                    "source_cacheables": cacheables,
                }
            ]

        summarizer._sliding_regions_for_doc = fake_regions
        summarizer._score_sliding_regions_vectorized = (
            lambda query_vector, regions, profile=None: [1.0 for _ in regions]
        )

        output = summarizer.compress_batch_top_k_docs([[doc]], ["query"])[0][0]

        self.assertEqual(
            [cacheable.text for cacheable in output.cacheables], ["one two three four"]
        )
        token_count = sum(
            summarizer._cacheable_token_len(cacheable)
            for cacheable in output.cacheables
        )
        self.assertGreaterEqual(token_count, summarizer.final_token_budget)

    def test_colbert_ratio_budget_deduplicates_prompt_visible_cacheables(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.retain_token_ratio = 0.5
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        summarizer.artifact = FakeSidecarCountingArtifact()
        shared_first = CacheableChunk(id="shared", text="one two")
        shared_second = CacheableChunk(id="shared", text="one two")
        unique = CacheableChunk(id="unique", text="three four five")
        docs = [
            RetrievableChunk(id="doc0::ret_0", text="", cacheables=[shared_first]),
            RetrievableChunk(
                id="doc0::ret_1", text="", cacheables=[shared_second, unique]
            ),
        ]

        self.assertEqual(summarizer._resolve_final_token_budget(docs), 3)

    def test_colbert_overlapping_regions_deduplicate_text_and_budget(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 180
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        cacheables = [
            CacheableChunk(id="doc0::sent_0", text="alpha", chunk_start=0, chunk_end=5),
            CacheableChunk(id="doc0::sent_1", text="beta", chunk_start=6, chunk_end=10),
            CacheableChunk(
                id="doc0::sent_2", text="gamma", chunk_start=11, chunk_end=16
            ),
        ]
        first_region = {
            "chunk_idx": 0,
            "center_idx": 0,
            "region_id": "doc0::region_0_1",
            "parent_doc_id": "doc0",
            "selected_indices": (0, 1),
            "source_cacheables": cacheables,
        }
        second_region = {
            "chunk_idx": 0,
            "center_idx": 1,
            "region_id": "doc0::region_1_2",
            "parent_doc_id": "doc0",
            "selected_indices": (1, 2),
            "source_cacheables": cacheables,
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, first_region), (9.0, second_region)],
            final_token_budget=3,
        )
        output_doc = ColBERTSlidingRegionCompressor._build_region_document(
            RetrievableChunk(id="doc0", text="alpha beta gamma", cacheables=cacheables),
            [cacheable for _, cacheable in selected],
        )

        self.assertEqual(
            [cacheable.sentence_ids for cacheable in output_doc.cacheables],
            [["doc0::sent_0", "doc0::sent_1"], ["doc0::sent_2"]],
        )
        self.assertEqual(
            [cacheable.text for cacheable in output_doc.cacheables],
            ["alpha beta", "gamma"],
        )
        self.assertEqual(
            sum(
                summarizer._cacheable_token_len(cacheable) for _, cacheable in selected
            ),
            3,
        )

    def test_colbert_cross_boundary_sentence_is_not_duplicated_or_double_charged(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        summarizer.region_token_budget = 180
        summarizer._token_len_cache = {}
        summarizer.budget_tokenizer = FakeBudgetTokenizer()
        left_sentence = CacheableChunk(
            id="doc0::sent_0", text="boundary evidence", chunk_start=0, chunk_end=17
        )
        right_sentence = CacheableChunk(
            id="doc0::sent_0", text="boundary evidence", chunk_start=0, chunk_end=17
        )
        left_region = {
            "chunk_idx": 0,
            "center_idx": 0,
            "region_id": "doc0::ret_0::region",
            "parent_doc_id": "doc0::ret_0",
            "selected_indices": (0,),
            "source_cacheables": [left_sentence],
        }
        right_region = {
            "chunk_idx": 1,
            "center_idx": 0,
            "region_id": "doc0::ret_1::region",
            "parent_doc_id": "doc0::ret_1",
            "selected_indices": (0,),
            "source_cacheables": [right_sentence],
        }

        selected = summarizer._select_sliding_regions(
            [(10.0, left_region), (9.0, right_region)],
            final_token_budget=4,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1].text, "boundary evidence")

    def test_subchunk_uses_budget_state_while_rerank_skips_it(self):
        FakeQueryEncoder.init_kwargs.clear()

        class FakeArtifact:
            index = {
                "checkpoint_name": "colbert-ir/colbertv2.0",
                "source_tokenizer_name": "artifact-tokenizer",
                "official_doc_maxlen": 180,
                "official_query_maxlen": 32,
            }
            retrievable_vectors_cache = {}

            def __init__(self, artifact_dir):
                self.artifact_dir = artifact_dir

        calls = []

        def fake_from_pretrained(name):
            calls.append(name)
            return FakeBudgetTokenizer()

        env = {
            "COLBERT_WINDOW_DIR": "/tmp/fake-colbert-window",
            "COLBERT_MODEL_NAME": "colbert-ir/colbertv2.0",
            "COLBERT_RERANK_KEEP": "2",
            "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct",
            "RETAIN_TOKEN_RATIO": "0.5",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch(
                "compressor.methods.colbert.base.ColBERTWindowArtifact",
                FakeArtifact,
            ),
            patch(
                "compressor.methods.colbert.base.ColBERTEncoder",
                FakeQueryEncoder,
            ),
            patch(
                "compressor.token_budget.AutoTokenizer.from_pretrained",
                side_effect=fake_from_pretrained,
            ),
        ):
            subchunk = ColBERTSubchunkCompressor()
            rerank = ColBERTRerankCompressor()

        self.assertEqual(
            calls,
            ["meta-llama/Llama-3.2-1B-Instruct"],
        )
        self.assertEqual(subchunk.retain_token_ratio, 0.5)
        self.assertTrue(hasattr(subchunk, "budget_tokenizer"))
        self.assertTrue(hasattr(subchunk, "_token_len_cache"))
        self.assertFalse(hasattr(rerank, "retain_token_ratio"))
        self.assertFalse(hasattr(rerank, "budget_tokenizer"))
        self.assertFalse(hasattr(rerank, "_token_len_cache"))
        self.assertTrue(FakeQueryEncoder.init_kwargs)
        self.assertTrue(
            all(call["device"] == "cpu" for call in FakeQueryEncoder.init_kwargs)
        )
        self.assertTrue(
            all(call["query_maxlen"] == 32 for call in FakeQueryEncoder.init_kwargs)
        )
        self.assertTrue(
            all(call["query_minlen"] is None for call in FakeQueryEncoder.init_kwargs)
        )
        self.assertTrue(
            all(
                call["query_truncation_side"] == "right"
                for call in FakeQueryEncoder.init_kwargs
            )
        )

    def test_colbert_query_ablation_overrides_maxlen_and_truncation_side(self):
        FakeQueryEncoder.init_kwargs.clear()

        class FakeArtifact:
            index = {
                "checkpoint_name": "colbert-ir/colbertv2.0",
                "official_query_maxlen": 32,
            }
            retrievable_vectors_cache = {}

            def __init__(self, artifact_dir):
                self.artifact_dir = artifact_dir

        env = {
            "COLBERT_WINDOW_DIR": "/tmp/fake-colbert-window",
            "COLBERT_MODEL_NAME": "colbert-ir/colbertv2.0",
            "COLBERT_QUERY_MAXLEN": "100",
            "COLBERT_QUERY_MINLEN": "32",
            "COLBERT_QUERY_TRUNCATION_SIDE": "left",
            "COLBERT_RERANK_KEEP": "2",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "compressor.methods.colbert.base.ColBERTWindowArtifact",
                FakeArtifact,
            ),
            patch(
                "compressor.methods.colbert.base.ColBERTEncoder",
                FakeQueryEncoder,
            ),
        ):
            ColBERTRerankCompressor()

        self.assertEqual(FakeQueryEncoder.init_kwargs[-1]["query_maxlen"], 100)
        self.assertEqual(FakeQueryEncoder.init_kwargs[-1]["query_minlen"], 32)
        self.assertEqual(
            FakeQueryEncoder.init_kwargs[-1]["query_truncation_side"], "left"
        )

    def test_colbert_subchunk_ratio_budget_controls_selection(self):
        summarizer = object.__new__(ColBERTSubchunkCompressor)
        summarizer.retain_token_ratio = 0.5
        summarizer.encoder = FakeQueryEncoder()
        summarizer._cacheable_token_lens = lambda cacheables: [
            cacheable.chunk_size for cacheable in cacheables
        ]
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        cacheables = [
            CacheableChunk(id="first", text="first", chunk_size=2),
            CacheableChunk(id="second", text="second", chunk_size=2),
        ]
        doc = RetrievableChunk(id="doc", text="first second", cacheables=cacheables)
        candidates = [
            {"chunk_idx": 0, "cacheable_idx": 0, "cacheable_id": "first"},
            {"chunk_idx": 0, "cacheable_idx": 1, "cacheable_id": "second"},
        ]
        summarizer._iter_candidates = lambda docs: candidates
        summarizer._score_candidate = (
            lambda query_vector, candidate: 2.0 - candidate["cacheable_idx"]
        )

        output = summarizer.compress_batch_top_k_docs([[doc]], ["query"])[0][0]

        self.assertEqual([cacheable.id for cacheable in output.cacheables], ["first"])

    def test_colbert_subchunk_adds_last_candidate_before_budget_check(self):
        summarizer = object.__new__(ColBERTSubchunkCompressor)
        summarizer.retain_token_ratio = None
        summarizer.final_token_budget = 3
        summarizer.encoder = FakeQueryEncoder()
        summarizer._cacheable_token_len = lambda cacheable: cacheable.chunk_size
        cacheables = [
            CacheableChunk(id="first", text="first", chunk_size=2),
            CacheableChunk(id="second", text="second", chunk_size=2),
        ]
        doc = RetrievableChunk(id="doc", text="first second", cacheables=cacheables)
        candidates = [
            {"chunk_idx": 0, "cacheable_idx": 0, "cacheable_id": "first"},
            {"chunk_idx": 0, "cacheable_idx": 1, "cacheable_id": "second"},
        ]
        summarizer._iter_candidates = lambda docs: candidates
        summarizer._score_candidate = (
            lambda query_vector, candidate: 2.0 - candidate["cacheable_idx"]
        )

        output = summarizer.compress_batch_top_k_docs([[doc]], ["query"])[0][0]

        self.assertEqual(
            [cacheable.id for cacheable in output.cacheables], ["first", "second"]
        )

    def test_colbert_budget_text_matches_prompt_processor_passage_format(self):
        summarizer = object.__new__(ColBERTSlidingRegionCompressor)
        cacheable = CacheableChunk(id="doc0::sent_0", text="  alpha beta  ")
        prompt_processor = PromptProcessor(
            tokenizer=object(),
            system_prompt="system",
            passage_prefix="",
        )

        self.assertEqual(
            summarizer._format_budget_cacheable_text(cacheable),
            prompt_processor.format_passage_chunk(cacheable.text),
        )

    def test_configured_retrieval_chunk_size_uses_db_manifest(self):
        with (
            patch.dict(os.environ, {"DB_DIR": "/fake/db"}, clear=True),
            patch(
                "compressor.methods.colbert.base.read_db_build_manifest",
                return_value={"retrievable_chunk_size": 512},
            ),
        ):
            self.assertEqual(_resolve_configured_retrieval_chunk_size(), 512)

    def test_configured_retrieval_chunk_size_requires_db_manifest(self):
        with (
            patch.dict(os.environ, {"DB_DIR": "/fake/db"}, clear=True),
            patch(
                "compressor.methods.colbert.base.read_db_build_manifest",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "manifest is required"):
                _resolve_configured_retrieval_chunk_size()


if __name__ == "__main__":
    unittest.main()
