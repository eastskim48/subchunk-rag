"""Document preprocessing pipeline for chunks and vector DB input."""

import os
import hashlib
import re
from tqdm import tqdm
import time
from transformers import AutoTokenizer

from vectordb import VectorDB
from materialize.splitter import SentenceWiseSplitter
from subchunk_store import SubchunkWriter


class DocumentPreprocessor:
    """Coordinate sentence splitting and retrieval DB writes."""

    def __init__(
        self,
        docs_dir: str,
        vectordb: VectorDB | None,
        model_name: str,
        cacheable_chunk_size: int | None = None,
        retrievable_chunk_size: int | None = None,
        db_batch_size: int = 256,
        materialize_db: bool = True,
        deduplicate_documents_by_hash: bool = False,
        max_subchunk_tokens: int | None = None,
    ):
        self.docs_dir = docs_dir
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.db_batch_size = db_batch_size
        self.materialize_db = materialize_db
        self.deduplicate_documents_by_hash = deduplicate_documents_by_hash
        self.max_subchunk_tokens = max_subchunk_tokens
        self._seen_document_hashes: dict[str, str] = {}
        self.duplicate_document_count = 0
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")

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

        if self.materialize_db and pending_db_chunks:
            assert self.vectordb is not None
            self.vectordb.store(pending_db_chunks)

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
