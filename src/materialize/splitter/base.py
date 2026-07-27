"""Compose parsers and groupers into retrievable/cacheable document chunks."""

import os
from dataclasses import dataclass
from typing import List

from chunk import RetrievableChunk, CacheableChunk
from materialize.splitter.grouper import IdentityGrouper, UnitGrouper
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
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        max_subchunk_tokens: int | None = None,
        parser: UnitParser | None = None,
    ):
        self.docs_dir = docs_dir
        self.model = model
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.content_chunk_size = content_chunk_size
        self.max_subchunk_tokens = max_subchunk_tokens
        self.parser = parser
        self.prompt_tokenizer_name = getattr(model, "model_name", None) or getattr(
            model.tokenizer, "name_or_path", None
        )
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
        token_ids = self.model.tokenizer.encode(text, add_special_tokens=False)
        return text, token_ids

    def _prompt_visible_token_count(self, text: str) -> int:
        prompt_text = f"{text.strip()}\n\n"
        return len(
            self.model.tokenizer(
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

            window_text = self.model.tokenizer.decode(
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


class FixedSizeSplitter(DocumentSplitter):
    """Create fixed-token cacheable units without semantic parsing."""

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        chunks = []
        chunk_starts = []
        chunk_ends = []
        max_chunk_tokens = 0
        for i in range(0, len(token_ids), self.content_chunk_size):
            chunk_tokens = token_ids[i : i + self.content_chunk_size]
            max_chunk_tokens = max(max_chunk_tokens, len(chunk_tokens))
            chunk_starts.append(i)
            chunk_ends.append(i + len(chunk_tokens))
            chunk_text = self.model.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True
            )
            chunk_id = f"{filename}-{i}"
            chunks.append(
                CacheableChunk(
                    id=chunk_id,
                    text=chunk_text,
                    parent_doc_id=filename,
                    chunk_size=self.cacheable_chunk_size,
                    chunk_start=i,
                    chunk_end=i + len(chunk_tokens),
                    prompt_token_count=self._prompt_visible_token_count(chunk_text),
                    prompt_tokenizer_name=self.prompt_tokenizer_name,
                )
            )
        return chunks, chunk_starts, chunk_ends, max_chunk_tokens


# Parsed-unit splitters follow a parser -> grouper pipeline:
# the parser extracts ordered units with source spans, and the grouper returns
# unit-index groups that this splitter materializes as cacheable chunks.
class ParsedUnitSplitter(DocumentSplitter):
    """Run a parser, then group its units into cacheable candidates."""

    def __init__(
        self,
        docs_dir,
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        parser: UnitParser,
        grouper: UnitGrouper,
        max_subchunk_tokens: int | None = None,
        split_long_units: bool = False,
        sentence_chunk_ids: bool = False,
    ):
        super().__init__(
            docs_dir=docs_dir,
            model=model,
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            content_chunk_size=content_chunk_size,
            max_subchunk_tokens=max_subchunk_tokens,
            parser=parser,
        )
        self.grouper = grouper
        self.split_long_units = split_long_units
        self.sentence_chunk_ids = sentence_chunk_ids

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        units = self.parser.parse(text, token_ids)
        if self.split_long_units:
            units = self.parser.split_long_units(
                units,
                token_ids,
                max_unit_tokens=self.max_subchunk_tokens,
            )
        unit_groups = self.grouper.group(units)

        chunks = []
        chunk_starts = []
        chunk_ends = []
        max_chunk_tokens = 0

        for group_idx, unit_indices in enumerate(unit_groups):
            unit_texts = [units[idx].text.strip() for idx in unit_indices]
            chunk_text = " ".join(unit_texts).strip()
            chunk_token_ids = self.model.tokenizer.encode(
                chunk_text, add_special_tokens=False
            )
            if not chunk_token_ids:
                continue

            first_unit = units[unit_indices[0]]
            last_unit = units[unit_indices[-1]]
            chunk_start = first_unit.token_start
            chunk_end = last_unit.token_end
            max_chunk_tokens = max(max_chunk_tokens, len(chunk_token_ids))
            chunk_starts.append(chunk_start)
            chunk_ends.append(chunk_end)
            sentence_ids = [f"{filename}::sent_{idx}" for idx in unit_indices]
            if self.sentence_chunk_ids:
                if len(unit_indices) != 1:
                    raise ValueError(
                        "sentence chunk IDs require one parsed unit per group"
                    )
                chunk_id = sentence_ids[0]
            else:
                chunk_id = f"{filename}::{self.grouper.name}::subchunk_{group_idx}"
            chunks.append(
                CacheableChunk(
                    id=chunk_id,
                    text=chunk_text,
                    parent_doc_id=filename,
                    chunk_size=self.cacheable_chunk_size,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    sentence_ids=sentence_ids,
                    sentence_texts=unit_texts,
                    prompt_token_count=self._prompt_visible_token_count(chunk_text),
                    prompt_tokenizer_name=self.prompt_tokenizer_name,
                )
            )

        return chunks, chunk_starts, chunk_ends, max_chunk_tokens


class SentenceWiseSplitter(ParsedUnitSplitter):
    """Use sentence parsing with identity grouping."""

    def __init__(
        self,
        docs_dir,
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        max_subchunk_tokens: int | None = None,
        parser: SentenceParser | None = None,
    ):
        super().__init__(
            docs_dir=docs_dir,
            model=model,
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            content_chunk_size=content_chunk_size,
            max_subchunk_tokens=max_subchunk_tokens,
            parser=(parser if parser is not None else SentenceParser(model.tokenizer)),
            grouper=IdentityGrouper(),
            split_long_units=True,
            sentence_chunk_ids=True,
        )


class SemanticSplitter(ParsedUnitSplitter):
    """Use sentence parsing with a configurable semantic grouper."""

    def __init__(
        self,
        docs_dir,
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        grouper: UnitGrouper,
        parser: UnitParser | None = None,
    ):
        super().__init__(
            docs_dir=docs_dir,
            model=model,
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            content_chunk_size=content_chunk_size,
            parser=(parser if parser is not None else SentenceParser(model.tokenizer)),
            grouper=grouper,
        )
