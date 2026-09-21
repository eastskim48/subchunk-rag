"""Build sentence cacheables and fixed-window retrieval chunks."""

import os
from dataclasses import dataclass
from typing import List

from chunk import RetrievableChunk, CacheableChunk
from materialize.splitter.parser import SentenceParser, UnitParser


@dataclass
class SplitDocumentResult:
    """Retriever chunks plus their fine-grained cacheable candidates."""

    chunks: List[CacheableChunk]
    retrievable_chunk: RetrievableChunk | None
    retrievable_chunks: List[RetrievableChunk] | None
    token_count: int
    max_chunk_tokens: int


class DocumentSplitter:
    """Base document loader and retrievable-chunk assembly pipeline."""

    def __init__(
        self,
        docs_dir,
        tokenizer,
        cacheable_chunk_size,
        retrievable_chunk_size,
        max_subchunk_tokens: int | None = None,
        parser: UnitParser | None = None,
    ):
        self.docs_dir = docs_dir
        self.tokenizer = tokenizer
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.max_subchunk_tokens = max_subchunk_tokens
        self.parser = parser
        if self.max_subchunk_tokens is not None and self.max_subchunk_tokens <= 0:
            raise ValueError(
                f"max_subchunk_tokens must be positive, got {self.max_subchunk_tokens}"
            )

    def split_document(self, filename: str) -> SplitDocumentResult:
        text, token_ids = self._load_source_document(filename)
        token_count = len(token_ids)
        chunks, _, _, max_chunk_tokens = self.build_chunks(
            filename,
            text,
            token_ids,
        )

        if not chunks:
            return SplitDocumentResult(
                chunks=[],
                retrievable_chunk=None,
                retrievable_chunks=[],
                token_count=token_count,
                max_chunk_tokens=0,
            )

        if self.retrievable_chunk_size is None:
            retrievable_chunks = [
                RetrievableChunk(
                    id=filename,
                    text=text,
                    cacheables=chunks,
                    chunk_size=self.cacheable_chunk_size,
                    token_count=token_count,
                )
            ]
        else:
            retrievable_chunks = self._build_retrievable_chunks(
                filename=filename,
                token_ids=token_ids,
                cacheables=chunks,
            )
        return SplitDocumentResult(
            chunks=chunks,
            retrievable_chunk=(
                retrievable_chunks[0] if len(retrievable_chunks) == 1 else None
            ),
            retrievable_chunks=retrievable_chunks,
            token_count=token_count,
            max_chunk_tokens=max_chunk_tokens,
        )

    def _load_source_document(self, filename: str):
        with open(os.path.join(self.docs_dir, filename)) as f:
            text = f.read()
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text, token_ids

    def _prompt_visible_token_count(self, text: str) -> int:
        prompt_text = f"{text.strip()}\n\n"
        return len(
            self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )

    def _build_retrievable_chunks(
        self,
        filename: str,
        token_ids: List[int],
        cacheables: List[CacheableChunk],
    ) -> List[RetrievableChunk]:
        if self.retrievable_chunk_size is None:
            raise ValueError(
                "_build_retrievable_chunks requires retrievable_chunk_size to be set"
            )

        retrievable_chunks = []
        for window_idx, window_start in enumerate(
            range(0, len(token_ids), self.retrievable_chunk_size)
        ):
            window_end = min(window_start + self.retrievable_chunk_size, len(token_ids))
            window_token_ids = token_ids[window_start:window_end]
            overlapping_cacheables = [
                cacheable
                for cacheable in cacheables
                if cacheable.chunk_start is not None
                and cacheable.chunk_end is not None
                and cacheable.chunk_start < window_end
                and cacheable.chunk_end > window_start
            ]
            if not overlapping_cacheables:
                continue

            window_text = self.tokenizer.decode(
                window_token_ids, skip_special_tokens=True
            )
            retrievable_chunks.append(
                RetrievableChunk(
                    id=f"{filename}::ret_{window_idx}",
                    text=window_text,
                    cacheables=overlapping_cacheables,
                    chunk_size=self.retrievable_chunk_size,
                    token_count=len(window_token_ids),
                    metadata={
                        "parent_doc_id": filename,
                        "source_token_start": window_start,
                        "source_token_end": window_end,
                    },
                )
            )

        return retrievable_chunks

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        raise NotImplementedError


class SentenceWiseSplitter(DocumentSplitter):
    """Create one cacheable candidate per parsed sentence unit."""

    def __init__(
        self,
        docs_dir,
        tokenizer,
        cacheable_chunk_size,
        retrievable_chunk_size,
        max_subchunk_tokens: int | None = None,
        parser: SentenceParser | None = None,
    ):
        super().__init__(
            docs_dir=docs_dir,
            tokenizer=tokenizer,
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            max_subchunk_tokens=max_subchunk_tokens,
            parser=(parser if parser is not None else SentenceParser(tokenizer)),
        )

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        units = self.parser.parse(text, token_ids)
        units = self.parser.split_long_units(
            units,
            token_ids,
            max_unit_tokens=self.max_subchunk_tokens,
        )

        chunks = []
        chunk_starts = []
        chunk_ends = []
        max_chunk_tokens = 0
        for unit_idx, unit in enumerate(units):
            chunk_text = unit.text.strip()
            chunk_token_ids = self.tokenizer.encode(
                chunk_text, add_special_tokens=False
            )
            if not chunk_token_ids:
                continue

            chunk_id = f"{filename}::sent_{unit_idx}"
            max_chunk_tokens = max(max_chunk_tokens, len(chunk_token_ids))
            chunk_starts.append(unit.token_start)
            chunk_ends.append(unit.token_end)
            chunks.append(
                CacheableChunk(
                    id=chunk_id,
                    text=chunk_text,
                    parent_doc_id=filename,
                    chunk_start=unit.token_start,
                    chunk_end=unit.token_end,
                    sentence_ids=[chunk_id],
                    sentence_texts=[chunk_text],
                    prompt_token_count=self._prompt_visible_token_count(chunk_text),
                )
            )

        return chunks, chunk_starts, chunk_ends, max_chunk_tokens
