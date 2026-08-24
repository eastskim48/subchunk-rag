import sys
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entrypoint import preprocess
from entrypoint import materialize_candidate_store
from chunk import CacheableChunk
from materialize.materialize import DocumentPreprocessor
from vectordb import ChromaDB


class PreprocessEntrypointTest(unittest.TestCase):
    def test_runtime_chromadb_constructor_exposes_no_embedding_device(self):
        parameter_names = set(inspect.signature(ChromaDB).parameters)

        self.assertEqual(parameter_names, {"db_dir"})

    def test_preprocess_interface_has_no_candidate_store_options(self):
        parameter_names = set(inspect.signature(preprocess.main).parameters)

        self.assertFalse(
            parameter_names
            & {
                "materialize_dense_embeds",
                "dense_embed_dir",
                "dense_embed_model",
                "dense_embed_overwrite",
            }
        )

    def test_candidate_store_interface_has_no_title_prefix_options(self):
        parameters = inspect.signature(materialize_candidate_store.main).parameters
        parameter_names = set(parameters)

        self.assertNotIn("prefix_title", parameter_names)
        self.assertNotIn("title_separator", parameter_names)
        self.assertNotIn("validate_against_db", parameter_names)
        self.assertEqual(parameters["center_unit"].default, "subchunk_only")

    def test_fixed_retrievable_chunk_preserves_source_span_metadata(self):
        cacheable = CacheableChunk(
            id="doc_0.txt-0",
            text="fixed chunk",
            parent_doc_id="doc_0.txt",
            chunk_size=128,
            chunk_start=0,
            chunk_end=127,
        )
        preprocessor = DocumentPreprocessor.__new__(DocumentPreprocessor)
        preprocessor.splitter = SimpleNamespace(
            split_document=lambda _: SimpleNamespace(
                token_count=127,
                chunks=[cacheable],
                max_chunk_tokens=127,
                retrievable_chunk=None,
                retrievable_chunks=[],
            )
        )
        preprocessor.total_doc_tokens = 0
        preprocessor.processed_doc_count = 0
        preprocessor.cacheable_chunk_size = 128
        preprocessor.docs_over_chunk_size = 0
        preprocessor.total_chunk_count = 0
        preprocessor.max_chunk_tokens = 0
        preprocessor.splitter_name = "fixed_size"

        _, retrievable_chunks = preprocessor.split_document("doc_0.txt")

        self.assertEqual(
            retrievable_chunks[0].metadata,
            {
                "parent_doc_id": "doc_0.txt",
                "source_token_start": 0,
                "source_token_end": 127,
            },
        )

    def test_materialize_db_false_does_not_open_chromadb(self):
        with (
            patch.object(preprocess, "TokenizerOnlyModel", return_value=object()),
            patch.object(preprocess, "ChromaDB") as chroma_db,
            patch.object(preprocess, "DocumentPreprocessor") as preprocessor,
        ):
            preprocess.main(
                docs_dir="documents",
                db_dir="db",
                cache_dir="cache",
                materialize_cache=False,
                materialize_db=False,
            )

        chroma_db.assert_not_called()
        self.assertIsNone(preprocessor.call_args.kwargs["vectordb"])
        preprocessor.return_value.process_documents.assert_called_once_with()

    def test_preprocess_passes_build_only_chroma_embedding_settings(self):
        with (
            patch.object(preprocess, "TokenizerOnlyModel", return_value=object()),
            patch.object(preprocess, "ChromaDB", return_value=object()) as chroma_db,
            patch.object(preprocess, "DocumentPreprocessor"),
            patch.object(preprocess, "build_db_manifest") as build_manifest,
            patch.object(preprocess, "write_db_build_manifest"),
        ):
            preprocess.main(
                docs_dir="documents",
                db_dir="db",
                cache_dir="cache",
                materialize_cache=False,
                materialize_db=True,
                chroma_embed_device="cuda",
                chroma_embed_batch_size=128,
            )

        chroma_db.for_build.assert_called_once_with(
            "db",
            embedding_device="cuda",
            embedding_batch_size=128,
        )
        self.assertEqual(build_manifest.call_args.kwargs["embedding_device"], "cuda")
        self.assertEqual(build_manifest.call_args.kwargs["embedding_batch_size"], 128)

    def test_db_chunks_are_upserted_in_batches_without_reordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filenames = ["doc_0.txt", "doc_1.txt", "doc_2.txt"]
            for filename in filenames:
                (Path(tmpdir) / filename).write_text("unused", encoding="utf-8")

            chunks_by_filename = {
                "doc_0.txt": ["r0", "r1", "r2"],
                "doc_1.txt": ["r3"],
                "doc_2.txt": ["r4"],
            }
            preprocessor = DocumentPreprocessor.__new__(DocumentPreprocessor)
            preprocessor.docs_dir = tmpdir
            preprocessor.deduplicate_documents_by_hash = False
            preprocessor.materialize_db = True
            preprocessor.materialize_cache = False
            preprocessor.db_batch_size = 2
            preprocessor.vectordb = Mock()
            preprocessor.split_document = lambda filename: (
                [f"cacheable-{filename}"],
                chunks_by_filename[filename],
            )
            preprocessor._should_skip_duplicate_document = lambda _: False
            preprocessor.processed_doc_count = 0
            preprocessor.total_chunk_count = 0
            preprocessor.resume_from_cache = False

            with patch(
                "materialize.materialize.os.listdir",
                return_value=filenames,
            ):
                preprocessor.process_documents()

            stored_batches = [
                call.args[0] for call in preprocessor.vectordb.store.call_args_list
            ]
            self.assertEqual(stored_batches, [["r0", "r1"], ["r2", "r3"], ["r4"]])

    def test_colbert_artifact_build_uses_fixed_cuda_policy(self):
        with (
            patch.object(
                materialize_candidate_store,
                "build_colbert_window_artifact",
                return_value={},
            ) as build_artifact,
            patch.object(
                materialize_candidate_store,
                "add_region_specs_to_colbert_window_data",
                return_value={},
            ),
            patch.object(
                materialize_candidate_store,
                "validate_colbert_candidate_ids_against_db",
                return_value={},
            ) as validate_candidate_ids,
        ):
            materialize_candidate_store.main(
                docs_dir="documents",
                output_dir="colbert_window",
                db_dir="db",
                model_name="checkpoint",
                db_batch_size=17,
            )

        build_kwargs = build_artifact.call_args.kwargs
        self.assertEqual(build_kwargs["model_name"], "checkpoint")
        self.assertEqual(build_kwargs["device"], "cuda")
        self.assertEqual(build_kwargs["db_batch_size"], 17)
        self.assertFalse(build_kwargs["verify_tensorization"])
        validate_candidate_ids.assert_called_once_with(
            artifact_dir="colbert_window",
            db_dir="db",
            batch_size=2048,
        )

    def test_dense_candidate_store_dispatches_without_colbert_build(self):
        with (
            patch.object(
                materialize_candidate_store,
                "build_dense_embedding_artifact_from_db",
                return_value={"documents": 1},
            ) as build_dense,
            patch.object(
                materialize_candidate_store, "build_colbert_window_artifact"
            ) as build_colbert,
        ):
            materialize_candidate_store.main(
                backend="dense",
                output_dir="dense_embed",
                db_dir="db",
                model_name="dense-checkpoint",
                batch_size=64,
                db_batch_size=17,
                overwrite=True,
            )

        build_dense.assert_called_once_with(
            db_dir="db",
            output_dir="dense_embed",
            embedding_model="dense-checkpoint",
            embedding_batch_size=64,
            db_batch_size=17,
            cache_unit=None,
            overwrite=True,
        )
        build_colbert.assert_not_called()

    def test_candidate_store_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "colbert.*dense"):
            materialize_candidate_store.main(
                backend="unknown",
                output_dir="artifact",
                db_dir="db",
            )


if __name__ == "__main__":
    unittest.main()
