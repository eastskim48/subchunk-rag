import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk
from colbert_metadata import ColBERTMetadataReader, ColBERTMetadataWriter
from materialize.colbert_materializer import (
    WindowSpec,
    _ColBERTArtifactWriter,
    _validate_center_unit_against_db_manifest,
    build_colbert_window_artifact,
    validate_colbert_candidate_ids_against_db,
)
from materialize.db_cacheables import group_unique_db_cacheables
from materialize.db_cacheables import iter_unique_db_cacheables_by_document


class DBCacheablesTest(unittest.TestCase):
    def test_grouping_deduplicates_overlap_and_preserves_source_order(self):
        later = CacheableChunk(
            id="doc.txt::sent_1",
            text="later",
            parent_doc_id="doc.txt",
            chunk_start=10,
            chunk_end=20,
        )
        earlier = CacheableChunk(
            id="doc.txt::sent_0",
            text="earlier",
            parent_doc_id="doc.txt",
            chunk_start=0,
            chunk_end=10,
        )

        grouped, stats = group_unique_db_cacheables([later, earlier, later.clone()])

        self.assertEqual(
            [cacheable.id for cacheable in grouped["doc.txt"]],
            ["doc.txt::sent_0", "doc.txt::sent_1"],
        )
        self.assertEqual(stats["db_cacheable_occurrences"], 3)
        self.assertEqual(stats["duplicate_cacheable_occurrences"], 1)
        self.assertEqual(stats["materialized_cacheables"], 2)

    def test_grouping_rejects_conflicting_duplicate_payloads(self):
        first = CacheableChunk(
            id="shared",
            text="first",
            parent_doc_id="doc.txt",
            chunk_start=0,
            chunk_end=1,
        )
        conflicting = CacheableChunk(
            id="shared",
            text="different",
            parent_doc_id="doc.txt",
            chunk_start=0,
            chunk_end=1,
        )

        with self.assertRaisesRegex(ValueError, "conflicting payloads"):
            group_unique_db_cacheables([first, conflicting])

    def test_streaming_grouping_deduplicates_and_preserves_source_order(self):
        later = CacheableChunk(
            id="doc.txt::sent_1",
            text="later",
            parent_doc_id="doc.txt",
            chunk_start=10,
            chunk_end=20,
        )
        earlier = CacheableChunk(
            id="doc.txt::sent_0",
            text="earlier",
            parent_doc_id="doc.txt",
            chunk_start=0,
            chunk_end=10,
        )

        with patch(
            "materialize.db_cacheables.iter_db_cacheables",
            return_value=iter([later, earlier, later.clone()]),
        ):
            groups = list(iter_unique_db_cacheables_by_document("db"))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], "doc.txt")
        self.assertEqual(
            [cacheable.id for cacheable in groups[0][1]],
            ["doc.txt::sent_0", "doc.txt::sent_1"],
        )

    def test_streaming_grouping_rejects_noncontiguous_parent_documents(self):
        cacheables = [
            CacheableChunk(id="a-0", text="a0", parent_doc_id="a"),
            CacheableChunk(id="b-0", text="b0", parent_doc_id="b"),
            CacheableChunk(id="a-1", text="a1", parent_doc_id="a"),
        ]

        with (
            patch(
                "materialize.db_cacheables.iter_db_cacheables",
                return_value=iter(cacheables),
            ),
            self.assertRaisesRegex(ValueError, "not contiguous"),
        ):
            list(iter_unique_db_cacheables_by_document("db"))

    def test_sentence_db_accepts_subchunk_center_units(self):
        manifest = {"splitter": "sentence", "cacheable_chunk_size": None}

        _validate_center_unit_against_db_manifest(manifest, "subchunk", None)
        _validate_center_unit_against_db_manifest(manifest, "subchunk_only", None)
        for removed_name in ("sentence", "sentence_only"):
            with self.subTest(removed_name=removed_name):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    _validate_center_unit_against_db_manifest(
                        manifest, removed_name, None
                    )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _validate_center_unit_against_db_manifest(
                manifest, "fixed_chunk", fixed_chunk_size=128
            )

    def test_fixed_db_requires_matching_chunk_size(self):
        manifest = {"splitter": "fixed_size", "cacheable_chunk_size": 128}

        _validate_center_unit_against_db_manifest(
            manifest, "fixed_chunk_window", fixed_chunk_size=128
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            _validate_center_unit_against_db_manifest(
                manifest, "fixed_chunk", fixed_chunk_size=256
            )

    def test_semantic_db_is_not_reinterpreted_as_sentence_units(self):
        with self.assertRaisesRegex(ValueError, "semantic"):
            _validate_center_unit_against_db_manifest(
                {"splitter": "semantic"}, "sentence", fixed_chunk_size=None
            )

    def test_candidate_id_validation_accepts_exact_db_alignment(self):
        cacheables = [
            CacheableChunk(id="first", text="one"),
            CacheableChunk(id="second", text="two"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.sqlite3"
            writer = ColBERTMetadataWriter(metadata_path)
            writer.add_cacheables([("first", 0, ["first"]), ("second", 1, ["second"])])
            writer.close()
            artifact = SimpleNamespace(
                data=SimpleNamespace(
                    id_to_row={"first": 0, "second": 1},
                    data_dir=Path(tmpdir),
                )
            )

            with (
                patch(
                    "materialize.colbert_materializer."
                    "colbert_artifact.ColBERTWindowArtifact",
                    return_value=artifact,
                ),
                patch(
                    "materialize.colbert_materializer.iter_db_cacheables",
                    return_value=iter(cacheables),
                ),
            ):
                summary = validate_colbert_candidate_ids_against_db(
                    artifact_dir="artifact",
                    db_dir="db",
                    batch_size=17,
                )

        self.assertEqual(summary["db_cacheable_count"], 2)
        self.assertEqual(summary["artifact_cacheable_count"], 2)
        self.assertEqual(summary["missing_in_artifact_count"], 0)
        self.assertEqual(summary["extra_in_artifact_count"], 0)

    def test_candidate_id_validation_rejects_missing_and_extra_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.sqlite3"
            writer = ColBERTMetadataWriter(metadata_path)
            writer.add_cacheables([("artifact-only", 0, ["artifact-only"])])
            writer.close()
            artifact = SimpleNamespace(
                data=SimpleNamespace(
                    id_to_row={"artifact-only": 0},
                    data_dir=Path(tmpdir),
                )
            )

            with (
                patch(
                    "materialize.colbert_materializer."
                    "colbert_artifact.ColBERTWindowArtifact",
                    return_value=artifact,
                ),
                patch(
                    "materialize.colbert_materializer.iter_db_cacheables",
                    return_value=iter([CacheableChunk(id="db-only", text="one")]),
                ),
                self.assertRaisesRegex(ValueError, "not aligned"),
            ):
                validate_colbert_candidate_ids_against_db(
                    artifact_dir="artifact",
                    db_dir="db",
                )

    def test_sentence_artifact_build_uses_db_cacheables_without_reading_documents(
        self,
    ):
        class FakeEncoder:
            captured_units = []
            encoded_specs = []

            def __init__(self, **kwargs):
                del kwargs
                self.max_length = 180
                self.doc_token_overhead = 3
                self.dim = 2
                self.device = "cpu"
                self.doc_maxlen = 180
                self.query_maxlen = 32
                self.checkpoint = SimpleNamespace(
                    colbert_config=SimpleNamespace(mask_punctuation=True)
                )

            def build_centered_windows(self, sentences, window_token_budget):
                self.captured_units.append((list(sentences), window_token_budget))
                return [
                    WindowSpec(
                        text=f"context::{text}",
                        center_start=0,
                        center_end=len(text),
                        selected_indices=[0, 1],
                        addition_order=[idx],
                        truncated_center=False,
                    )
                    for idx, text in enumerate(sentences)
                ]

            def encode_windows(self, specs):
                self.encoded_specs.extend(specs)
                return [torch.ones((1, self.dim), dtype=torch.float32) for _ in specs]

        cacheables = [
            CacheableChunk(
                id="doc.txt::sent_0",
                text="stored first",
                parent_doc_id="doc.txt",
                chunk_start=0,
                chunk_end=2,
            ),
            CacheableChunk(
                id="doc.txt::sent_1",
                text="stored second",
                parent_doc_id="doc.txt",
                chunk_start=2,
                chunk_end=4,
            ),
        ]
        manifest = {
            "splitter": "sentence",
            "tokenizer_name": "fake-tokenizer",
            "cacheable_chunk_size": None,
        }

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch(
                "materialize.colbert_materializer."
                "colbert_artifact.build_db_manifest_reference",
                return_value=(
                    manifest,
                    {"path": "../db/build_manifest.json", "sha256": "x" * 64},
                ),
            ),
            patch(
                "materialize.colbert_materializer."
                "iter_unique_db_cacheables_by_document",
                return_value=iter([("doc.txt", cacheables)]),
            ),
            patch(
                "materialize.colbert_materializer.AutoTokenizer.from_pretrained",
            ) as load_source_tokenizer,
            patch(
                "materialize.colbert_materializer.ColBERTWindowEncoder",
                FakeEncoder,
            ),
        ):
            summary = build_colbert_window_artifact(
                docs_dir="/path/that/does/not/exist",
                output_dir=output_dir,
                db_dir="/fake/db",
            )
            data_index = json.loads(
                (Path(output_dir) / "data" / "index.json").read_text(encoding="utf-8")
            )
            metadata_reader = ColBERTMetadataReader(
                Path(output_dir) / "data" / data_index["metadata_file"]
            )

        load_source_tokenizer.assert_not_called()
        self.assertEqual(
            FakeEncoder.captured_units,
            [(["stored first", "stored second"], 180)],
        )
        self.assertEqual(
            [spec.text for spec in FakeEncoder.encoded_specs],
            ["stored first", "stored second"],
        )
        self.assertEqual(
            [spec.selected_indices for spec in FakeEncoder.encoded_specs],
            [[0], [1]],
        )
        self.assertEqual(
            list(metadata_reader.iter_cacheable_ids()),
            ["doc.txt::sent_0", "doc.txt::sent_1"],
        )
        self.assertEqual(
            metadata_reader.window_ids_for_cacheable_ids(
                ["doc.txt::sent_0", "doc.txt::sent_1"]
            ),
            [
                ["doc.txt::sent_0", "doc.txt::sent_1"],
                ["doc.txt::sent_0", "doc.txt::sent_1"],
            ],
        )
        metadata_reader.close()
        self.assertEqual(summary["num_cacheables"], 2)
        self.assertEqual(summary["center_unit"], "subchunk_only")
        self.assertEqual(
            summary["artifact_variant"],
            "subchunk_only_encoding_contextual_regions",
        )

    def test_artifact_writer_carries_document_overflow_into_full_batches(self):
        class FakeEncoder:
            dim = 2

            def __init__(self):
                self.encoded_batches = []

            def encode_windows(self, specs):
                self.encoded_batches.append([spec.text for spec in specs])
                return [
                    torch.full((1, self.dim), index, dtype=torch.float32)
                    for index, _ in enumerate(specs)
                ]

        def specs(prefix, count):
            return [
                WindowSpec(
                    text=f"{prefix}{index}",
                    center_start=0,
                    center_end=2,
                    selected_indices=list(range(count)),
                    addition_order=[index],
                    truncated_center=False,
                )
                for index in range(count)
            ]

        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            writer = _ColBERTArtifactWriter(
                encoder=encoder,
                vectors_path=temp_path / "vectors.fp16.bin",
                metadata_path=temp_path / "metadata.sqlite3",
                batch_size=4,
            )
            first_specs = specs("a", 3)
            second_specs = specs("b", 2)
            third_specs = specs("c", 4)
            writer.add_document(
                "doc-a",
                [f"a{index}" for index in range(3)],
                first_specs,
            )
            writer.add_document(
                "doc-b",
                [f"b{index}" for index in range(2)],
                second_specs,
            )
            writer.add_document(
                "doc-c",
                [f"c{index}" for index in range(4)],
                third_specs,
            )
            writer.finalize()

            reader = ColBERTMetadataReader(temp_path / "metadata.sqlite3")
            try:
                self.assertEqual(
                    list(reader.iter_cacheable_ids()),
                    ["a0", "a1", "a2", "b0", "b1", "c0", "c1", "c2", "c3"],
                )
                self.assertEqual(
                    reader.window_ids_for_cacheable_ids(["b0"]),
                    [["b0", "b1"]],
                )
            finally:
                reader.close()

        self.assertEqual(
            encoder.encoded_batches,
            [
                ["a0", "a1", "a2", "b0"],
                ["b1", "c0", "c1", "c2"],
                ["c3"],
            ],
        )
        self.assertEqual(writer.num_docs, 3)
        self.assertEqual(writer.total_cacheables, 9)
        self.assertEqual(writer.total_center_tokens, 9)


if __name__ == "__main__":
    unittest.main()
