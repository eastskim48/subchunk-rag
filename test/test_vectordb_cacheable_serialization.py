import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chunk import CacheableChunk, RetrievableChunk
from vectordb import ChromaDB


class VectorDBCacheableSerializationTest(unittest.TestCase):
    @staticmethod
    def _cacheable() -> CacheableChunk:
        return CacheableChunk(
            id="doc.txt::sent_0",
            text="Stored sentence.",
            parent_doc_id="doc.txt",
            chunk_size=512,
            chunk_start=10,
            chunk_end=14,
            sentence_ids=["doc.txt::sent_0"],
            sentence_texts=["Stored sentence."],
            prompt_token_count=7,
            prompt_tokenizer_name="test-tokenizer",
        )

    def test_serialize_omits_only_unused_storage_fields(self):
        cacheable = self._cacheable()
        original_payload = cacheable.to_payload()

        serialized = ChromaDB._serialize_cacheables([cacheable])
        payload = json.loads(serialized)[0]

        expected_payload = dict(original_payload)
        expected_payload.pop("chunk_size")
        expected_payload.pop("sentence_texts")
        self.assertEqual(payload, expected_payload)
        self.assertNotIn("chunk_size", payload)
        self.assertNotIn("sentence_texts", payload)

    def test_serialize_does_not_mutate_cacheable_or_general_payload(self):
        cacheable = self._cacheable()
        original_payload = cacheable.to_payload()

        ChromaDB._serialize_cacheables([cacheable])

        self.assertEqual(cacheable.chunk_size, 512)
        self.assertEqual(cacheable.sentence_texts, ["Stored sentence."])
        self.assertEqual(cacheable.to_payload(), original_payload)

    def test_new_storage_payload_deserializes_with_defaults(self):
        cacheable = self._cacheable()

        restored = ChromaDB._deserialize_cacheables(
            ChromaDB._serialize_cacheables([cacheable])
        )[0]

        self.assertEqual(restored.id, cacheable.id)
        self.assertEqual(restored.text, cacheable.text)
        self.assertEqual(restored.parent_doc_id, cacheable.parent_doc_id)
        self.assertIsNone(restored.chunk_size)
        self.assertEqual(restored.chunk_start, cacheable.chunk_start)
        self.assertEqual(restored.chunk_end, cacheable.chunk_end)
        self.assertEqual(restored.sentence_ids, cacheable.sentence_ids)
        self.assertEqual(restored.sentence_texts, [])
        self.assertEqual(restored.prompt_token_count, cacheable.prompt_token_count)
        self.assertEqual(
            restored.prompt_tokenizer_name,
            cacheable.prompt_tokenizer_name,
        )

    def test_legacy_storage_payload_still_deserializes_removed_fields(self):
        cacheable = self._cacheable()
        legacy_value = json.dumps([cacheable.to_payload()])

        restored = ChromaDB._deserialize_cacheables(legacy_value)[0]

        self.assertEqual(restored.chunk_size, 512)
        self.assertEqual(restored.sentence_texts, ["Stored sentence."])

    def test_store_preserves_top_level_retrievable_chunk_size(self):
        cacheable = self._cacheable()
        retrievable = RetrievableChunk(
            id="doc.txt::ret_0",
            text="Retrievable text.",
            cacheables=[cacheable],
            chunk_size=512,
            token_count=14,
            cache_unit="sentence",
            metadata={"parent_doc_id": "doc.txt"},
        )
        vector_db = ChromaDB.__new__(ChromaDB)
        vector_db.db = Mock()

        vector_db.store([retrievable])

        metadata = vector_db.db.upsert.call_args.kwargs["metadatas"][0]
        nested_payload = json.loads(metadata["cacheables_json"])[0]
        self.assertEqual(metadata["chunk_size"], 512)
        self.assertEqual(metadata["token_count"], 14)
        self.assertEqual(metadata["cache_unit"], "sentence")
        self.assertEqual(metadata["parent_doc_id"], "doc.txt")
        self.assertNotIn("chunk_size", nested_payload)
        self.assertNotIn("sentence_texts", nested_payload)

    def test_empty_cacheable_list_round_trips(self):
        serialized = ChromaDB._serialize_cacheables([])

        self.assertEqual(json.loads(serialized), [])
        self.assertEqual(ChromaDB._deserialize_cacheables(serialized), [])


if __name__ == "__main__":
    unittest.main()
