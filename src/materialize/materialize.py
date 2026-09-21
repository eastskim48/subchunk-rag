"""Document preprocessing pipeline for chunks, KV caches, and vector DB input."""

import os
import hashlib
import re
import torch
from tqdm import tqdm
from typing import List
import time
from transformers import AutoTokenizer

from vectordb import VectorDB
from model import LLMModel
from chunk import Chunk, CacheableChunk
from materialize.splitter import SentenceWiseSplitter
from subchunk_store import SubchunkWriter


class DocumentPreprocessor:
    """Coordinate splitting, cache materialization, resume checks, and DB writes."""

    def __init__(
        self,
        docs_dir: str,
        vectordb: VectorDB | None,
        model_name: str,
        cache_dir: str,
        cacheable_chunk_size: int | None = None,
        retrievable_chunk_size: int | None = None,
        batch_size: int = 1,
        db_batch_size: int = 256,
        dummy_bos_count: int = 0,
        materialize_cache: bool = True,
        materialize_db: bool = True,
        resume_from_cache: bool = False,
        deduplicate_documents_by_hash: bool = False,
        max_subchunk_tokens: int | None = None,
    ):
        self.docs_dir = docs_dir
        self.cache_dir = cache_dir
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.batch_size = batch_size
        self.db_batch_size = db_batch_size
        self.dummy_bos_count = dummy_bos_count
        self.materialize_cache = materialize_cache
        self.materialize_db = materialize_db
        self.resume_from_cache = resume_from_cache
        self.deduplicate_documents_by_hash = deduplicate_documents_by_hash
        self.max_subchunk_tokens = max_subchunk_tokens
        self._seen_document_hashes: dict[str, str] = {}
        self.duplicate_document_count = 0
        if self.materialize_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.vectordb = vectordb
        if self.materialize_db and self.vectordb is None:
            raise ValueError("vectordb must be set when materialize_db=True")
        if self.db_batch_size <= 0:
            raise ValueError("db_batch_size must be positive")
        self.total_doc_tokens = 0
        self.processed_doc_count = 0
        self.docs_over_chunk_size = 0
        self.total_chunk_count = 0
        self.max_chunk_tokens = 0
        self.skipped_existing_chunk_count = 0
        self.rebuilt_invalid_chunk_count = 0

        if materialize_cache:
            self.model = LLMModel(model_name)
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="right"
            )

        if self.dummy_bos_count > 0 and self.tokenizer.bos_token_id is None:
            raise ValueError(
                "dummy_bos_count requires tokenizer.bos_token_id to be set"
            )
        self.visible_suffix_token_ids = self.tokenizer.encode(
            "\n\n",
            add_special_tokens=False,
        )
        if (
            self.cacheable_chunk_size is not None
            and self.retrievable_chunk_size is None
        ):
            raise NotImplementedError(
                "cacheable_chunk_size set with retrievable_chunk_size=None is not supported"
            )

        self.splitter = SentenceWiseSplitter(
            docs_dir=self.docs_dir,
            tokenizer=self.tokenizer,
            cacheable_chunk_size=self.cacheable_chunk_size,
            retrievable_chunk_size=self.retrievable_chunk_size,
            max_subchunk_tokens=self.max_subchunk_tokens,
        )

    def process_documents(self):
        if not self.materialize_db:
            return self._process_documents()
        if self.vectordb.db.count():
            raise ValueError(
                "integer subchunk assignment requires an empty retrieval DB"
            )
        writer = SubchunkWriter(self.vectordb.db_dir)
        self.subchunk_writer = writer
        try:
            result = self._process_documents()
            writer.finish()
            return result
        finally:
            writer.close()
            self.subchunk_writer = None

    def _process_documents(self):
        start_time = time.time()
        files = os.listdir(self.docs_dir)
        if self.deduplicate_documents_by_hash:
            files = sorted(files, key=self._document_sort_key)
        print(f"Processing {len(files)} documents...")
        pending_cache_chunks = []
        pending_db_chunks = []

        for filename in tqdm(files):
            if self._should_skip_duplicate_document(filename):
                continue
            cacheable_chunks, retrievable_chunks = self.split_document(filename)
            if not cacheable_chunks:
                continue
            if self.materialize_db:
                self.subchunk_writer.add_document(cacheable_chunks)
                assert self.vectordb is not None
                pending_db_chunks.extend(retrievable_chunks)
                while len(pending_db_chunks) >= self.db_batch_size:
                    current_batch = pending_db_chunks[: self.db_batch_size]
                    self.vectordb.store(current_batch)
                    pending_db_chunks = pending_db_chunks[self.db_batch_size :]
            if self.materialize_cache:
                if self.resume_from_cache:
                    cacheable_chunks = self._filter_chunks_for_resume(cacheable_chunks)
                    if not cacheable_chunks:
                        continue
                pending_cache_chunks.extend(cacheable_chunks)
                while len(pending_cache_chunks) >= self.batch_size:
                    current_batch = pending_cache_chunks[: self.batch_size]
                    self.save_kv_cache(current_batch)
                    pending_cache_chunks = pending_cache_chunks[self.batch_size :]

        if self.materialize_db and pending_db_chunks:
            assert self.vectordb is not None
            self.vectordb.store(pending_db_chunks)
        if self.materialize_cache and pending_cache_chunks:
            self.save_kv_cache(pending_cache_chunks)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.processed_doc_count > 0:
            avg_doc_tokens = self.total_doc_tokens / self.processed_doc_count
            print(f"Average document length: {avg_doc_tokens:.2f} tokens.")
            if self.cacheable_chunk_size is not None:
                print(
                    f"Documents longer than cacheable_chunk_size ({self.cacheable_chunk_size}): {self.docs_over_chunk_size}"
                )
        if self.total_chunk_count > 0:
            avg_chunks = self.total_chunk_count / self.processed_doc_count
            print(f"Average chunks per document: {avg_chunks:.2f}.")
            print(f"Max chunk length observed: {self.max_chunk_tokens} tokens.")
        if self.resume_from_cache:
            print(f"Skipped existing valid chunks: {self.skipped_existing_chunk_count}")
            print(
                f"Rebuilt invalid existing chunks: {self.rebuilt_invalid_chunk_count}"
            )
        if self.deduplicate_documents_by_hash:
            print(
                "Skipped duplicate documents by normalized text hash: "
                f"{self.duplicate_document_count}"
            )

        print(f"Processing completed in {elapsed_time:.2f} seconds.")

    @staticmethod
    def _normalized_document_text_for_hash(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _document_sort_key(filename: str):
        match = re.fullmatch(r"doc_(\d+)\.txt", filename)
        if match:
            return (0, int(match.group(1)), filename)
        return (1, filename)

    def _should_skip_duplicate_document(self, filename: str) -> bool:
        if not self.deduplicate_documents_by_hash:
            return False
        path = os.path.join(self.docs_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            normalized_text = self._normalized_document_text_for_hash(handle.read())
        digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        existing_filename = self._seen_document_hashes.get(digest)
        if existing_filename is not None:
            self.duplicate_document_count += 1
            return True
        self._seen_document_hashes[digest] = filename
        return False

    def _cache_path_for_chunk(self, chunk: Chunk) -> str:
        return os.path.join(self.cache_dir, f"{chunk.id}.pt")

    def _has_valid_existing_cache(self, chunk: Chunk) -> bool:
        cache_path = self._cache_path_for_chunk(chunk)
        if not os.path.exists(cache_path):
            return False
        try:
            torch.load(cache_path, weights_only=True, map_location="cpu")
            return True
        except Exception:
            try:
                os.remove(cache_path)
            except OSError:
                pass
            self.rebuilt_invalid_chunk_count += 1
            return False

    def _filter_chunks_for_resume(self, chunks: List[Chunk]) -> List[Chunk]:
        pending = []
        for chunk in chunks:
            if self._has_valid_existing_cache(chunk):
                self.skipped_existing_chunk_count += 1
                continue
            pending.append(chunk)
        return pending

    def split_document(self, filename: str):
        result = self.splitter.split_document(filename)
        self.total_doc_tokens += result.token_count
        self.processed_doc_count += 1
        if (
            self.cacheable_chunk_size is not None
            and result.token_count > self.cacheable_chunk_size
        ):
            self.docs_over_chunk_size += 1
        if result.chunks:  # cacheable chunk
            self.total_chunk_count += len(result.chunks)
            self.max_chunk_tokens = max(self.max_chunk_tokens, result.max_chunk_tokens)
        retrievable_chunks = list(
            result.retrievable_chunks
            or ([] if result.retrievable_chunk is None else [result.retrievable_chunk])
        )
        for retrievable_chunk in retrievable_chunks:
            if retrievable_chunk.cache_unit is None:
                retrievable_chunk.cache_unit = "sentence"
        return result.chunks, retrievable_chunks

    def _build_cache_inputs(self, chunks: List[CacheableChunk], padding_side: str):
        visible_sequences = [
            self._build_visible_chunk_token_ids(chunk) for chunk in chunks
        ]
        max_length = max(len(sequence) for sequence in visible_sequences)
        input_ids = []
        attention_masks = []
        pad_token_id = int(self.tokenizer.pad_token_id)
        for sequence in visible_sequences:
            pad_len = max_length - len(sequence)
            if padding_side == "left":
                padded_sequence = [pad_token_id] * pad_len + sequence
                attention_mask = [0] * pad_len + [1] * len(sequence)
            else:
                padded_sequence = sequence + [pad_token_id] * pad_len
                attention_mask = [1] * len(sequence) + [0] * pad_len
            input_ids.append(padded_sequence)
            attention_masks.append(attention_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        if self.dummy_bos_count == 0:
            return {
                "input_ids": input_ids.to("cuda"),
                "attention_mask": attention_mask.to("cuda"),
            }

        bos_prefix = torch.full(
            (input_ids.shape[0], self.dummy_bos_count),
            int(self.tokenizer.bos_token_id),
            dtype=input_ids.dtype,
        )
        bos_mask = torch.ones(
            (attention_mask.shape[0], self.dummy_bos_count),
            dtype=attention_mask.dtype,
        )
        input_ids = torch.cat([bos_prefix, input_ids], dim=1)
        attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
        return {
            "input_ids": input_ids.to("cuda"),
            "attention_mask": attention_mask.to("cuda"),
        }

    def _build_visible_chunk_token_ids(self, chunk: Chunk) -> List[int]:
        chunk_token_ids = self.tokenizer.encode(chunk.text, add_special_tokens=False)
        return chunk_token_ids + self.visible_suffix_token_ids

    def save_kv_cache(self, chunks: List[CacheableChunk]):
        inputs = self._build_cache_inputs(chunks, padding_side="right")
        with torch.no_grad():
            output = self.model.model(**inputs, use_cache=True)

        cache = output.past_key_values.to_legacy_cache()
        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1).tolist()

        for batch_idx, (chunk, sequence_length) in enumerate(
            zip(chunks, sequence_lengths)
        ):
            sample_cache = []
            seq_len = int(sequence_length)
            for layer in cache:
                key = layer[0][
                    batch_idx : batch_idx + 1, :, self.dummy_bos_count : seq_len, :
                ].contiguous()
                value = layer[1][
                    batch_idx : batch_idx + 1, :, self.dummy_bos_count : seq_len, :
                ].contiguous()
                sample_cache.append((key, value))
            torch.save(sample_cache, os.path.join(self.cache_dir, f"{chunk.id}.pt"))
