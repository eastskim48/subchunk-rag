import os
import re
from dataclasses import dataclass
from typing import List

from chunk import RetrievableChunk, CacheableChunk
from materialize.splitter.merger import SubchunkMerger


@dataclass
class SplitDocumentResult:
    chunks: List[CacheableChunk]
    retrievable_chunk: RetrievableChunk | None
    retrievable_chunks: List[RetrievableChunk] | None
    token_count: int
    max_chunk_tokens: int


@dataclass
class SentenceView:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


class DocumentSplitter:
    def __init__(
        self,
        docs_dir,
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        max_subchunk_tokens: int | None = None,
    ):
        self.docs_dir = docs_dir
        self.model = model
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.content_chunk_size = content_chunk_size
        self.max_subchunk_tokens = max_subchunk_tokens
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

    def _split_sentence_texts(self, text: str) -> List[str]:
        try:
            from blingfire import text_to_sentences
        except ImportError as exc:
            raise ImportError(
                "splitter='sentence' or splitter='semantic' requires blingfire to be installed"
            ) from exc

        sentence_texts = [
            part.strip()
            for part in text_to_sentences(text.strip()).splitlines()
            if part.strip()
        ]
        if not sentence_texts and text.strip():
            sentence_texts = [text.strip()]
        return self._merge_conservative_sentence_boundaries(sentence_texts)

    @staticmethod
    def _should_merge_with_next_sentence(
        prev_sentence: str, next_sentence: str
    ) -> bool:
        prev = prev_sentence.strip()
        nxt = next_sentence.strip()
        if not prev or not nxt:
            return False

        # Common abbreviation / title / legal-case fragments that should usually
        # stay attached to the following continuation.
        if re.search(
            r"(?:\b(?:v|vs|No|Mr|Mrs|Ms|Dr|Prof|Sr|Jr)|\b(?:U\.S|U\.K|D\.C))\.$", prev
        ):
            return True

        # Very short abbreviation-like fragments such as "A.", "J. K.", "Miller v."
        # are conservatively merged with the next segment.
        tokens = prev.split()
        if len(tokens) <= 3:
            if re.fullmatch(r"(?:[A-Z]\.\s*){1,3}", prev):
                return True
            if re.search(r"(?:\b[A-Z]\.|\b[a-z]\.)$", prev):
                return True

        return False

    def _merge_conservative_sentence_boundaries(
        self, sentence_texts: List[str]
    ) -> List[str]:
        if not sentence_texts:
            return sentence_texts

        merged: List[str] = []
        cursor = 0
        while cursor < len(sentence_texts):
            current = sentence_texts[cursor]
            while cursor + 1 < len(
                sentence_texts
            ) and self._should_merge_with_next_sentence(
                current,
                sentence_texts[cursor + 1],
            ):
                current = f"{current} {sentence_texts[cursor + 1]}".strip()
                cursor += 1
            merged.append(current)
            cursor += 1
        return merged

    def _locate_sentence_span(
        self, text: str, sentence_text: str, search_cursor: int
    ) -> tuple[int, int]:
        text_cursor = search_cursor
        while text_cursor < len(text) and text[text_cursor].isspace():
            text_cursor += 1
        if text.startswith(sentence_text, text_cursor):
            return text_cursor, text_cursor + len(sentence_text)

        normalized_sentence = " ".join(sentence_text.split())
        sentence_cursor = 0
        char_start = text_cursor

        while sentence_cursor < len(normalized_sentence):
            if text_cursor >= len(text):
                raise ValueError(
                    f"failed to locate sentence span in source document: {sentence_text!r}"
                )

            sentence_ch = normalized_sentence[sentence_cursor]
            text_ch = text[text_cursor]

            if sentence_ch.isspace():
                if not text_ch.isspace():
                    raise ValueError(
                        f"failed to locate sentence span in source document: {sentence_text!r}"
                    )
                while (
                    sentence_cursor < len(normalized_sentence)
                    and normalized_sentence[sentence_cursor].isspace()
                ):
                    sentence_cursor += 1
                while text_cursor < len(text) and text[text_cursor].isspace():
                    text_cursor += 1
                continue

            if text_ch.isspace():
                raise ValueError(
                    f"failed to locate sentence span in source document: {sentence_text!r}"
                )

            if text_ch != sentence_ch:
                raise ValueError(
                    f"failed to locate sentence span in source document: {sentence_text!r}"
                )

            sentence_cursor += 1
            text_cursor += 1

        char_end = text_cursor
        return char_start, char_end

    def _build_sentence_views(
        self, text: str, token_ids: List[int]
    ) -> List[SentenceView]:
        sentence_texts = self._split_sentence_texts(text)
        if not sentence_texts:
            return []

        encoded = self.model.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_token_ids = list(encoded["input_ids"])
        if full_token_ids != list(token_ids):
            token_ids = full_token_ids
        offsets = list(encoded["offset_mapping"])

        sentence_views = []
        search_cursor = 0
        token_cursor = 0
        for sentence_text in sentence_texts:
            try:
                char_start, char_end = self._locate_sentence_span(
                    text, sentence_text, search_cursor
                )
            except ValueError:
                # Some corpora (notably TriviaQA paste dumps) contain repeated,
                # poorly formatted QA blobs that make sentence-to-source span
                # rematching ambiguous mid-document. In that case, keep the
                # remaining tail as a single sentence view instead of aborting
                # the whole preprocess run.
                remaining_text = text[search_cursor:].strip()
                if not remaining_text:
                    break
                char_start = text.find(remaining_text, search_cursor)
                if char_start < 0:
                    raise
                char_end = char_start + len(remaining_text)
            search_cursor = char_end

            while (
                token_cursor < len(offsets) and offsets[token_cursor][1] <= char_start
            ):
                token_cursor += 1
            token_start = token_cursor
            while token_cursor < len(offsets) and offsets[token_cursor][0] < char_end:
                token_cursor += 1
            token_end = token_cursor

            sentence_views.append(
                SentenceView(
                    text=text[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                )
            )
            if char_end == len(text):
                break

        return sentence_views

    def _split_long_sentence_views(
        self, sentence_views: List[SentenceView], token_ids: List[int]
    ) -> List[SentenceView]:
        if self.max_subchunk_tokens is None:
            return sentence_views

        split_views: List[SentenceView] = []
        for view in sentence_views:
            token_length = view.token_end - view.token_start
            if token_length <= self.max_subchunk_tokens:
                split_views.append(view)
                continue

            part_start = view.token_start
            while part_start < view.token_end:
                part_end = min(part_start + self.max_subchunk_tokens, view.token_end)
                if part_end <= part_start:
                    raise ValueError(
                        "failed to split long sentence into a non-empty subchunk: "
                        f"token_start={view.token_start}, token_end={view.token_end}, "
                        f"part_start={part_start}, part_end={part_end}"
                    )
                part_tokens = token_ids[part_start:part_end]
                part_text = self.model.tokenizer.decode(
                    part_tokens, skip_special_tokens=True
                ).strip()
                if part_text:
                    split_views.append(
                        SentenceView(
                            text=part_text,
                            char_start=view.char_start,
                            char_end=view.char_end,
                            token_start=part_start,
                            token_end=part_end,
                        )
                    )
                part_start = part_end
        return split_views

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
                        "window_token_start": window_start,
                        "window_token_end": window_end,
                    },
                )
            )

        return retrievable_chunks

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        raise NotImplementedError


class FixedSizeSplitter(DocumentSplitter):
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


class SentenceWiseSplitter(DocumentSplitter):
    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        sentence_views = self._split_long_sentence_views(
            self._build_sentence_views(text, token_ids), token_ids
        )

        chunks = []
        chunk_starts = []
        chunk_ends = []
        max_chunk_tokens = 0
        for sent_idx, sentence_view in enumerate(sentence_views):
            sentence_text = sentence_view.text.strip()
            sentence_token_ids = self.model.tokenizer.encode(
                sentence_text, add_special_tokens=False
            )
            if not sentence_token_ids:
                continue
            chunk_start = sentence_view.token_start
            chunk_end = sentence_view.token_end
            max_chunk_tokens = max(max_chunk_tokens, len(sentence_token_ids))
            chunk_starts.append(chunk_start)
            chunk_ends.append(chunk_end)
            chunk_id = f"{filename}::sent_{sent_idx}"
            chunks.append(
                CacheableChunk(
                    id=chunk_id,
                    text=sentence_text,
                    parent_doc_id=filename,
                    chunk_size=self.cacheable_chunk_size,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    sentence_ids=[chunk_id],
                    sentence_texts=[sentence_text],
                    prompt_token_count=self._prompt_visible_token_count(sentence_text),
                    prompt_tokenizer_name=self.prompt_tokenizer_name,
                )
            )

        return chunks, chunk_starts, chunk_ends, max_chunk_tokens


class SemanticSplitter(DocumentSplitter):
    def __init__(
        self,
        docs_dir,
        model,
        cacheable_chunk_size,
        retrievable_chunk_size,
        content_chunk_size,
        merger: SubchunkMerger,
    ):
        super().__init__(
            docs_dir,
            model,
            cacheable_chunk_size,
            retrievable_chunk_size,
            content_chunk_size,
        )
        self.merger = merger

    def build_chunks(self, filename: str, text: str, token_ids: List[int]):
        sentence_views = self._build_sentence_views(text, token_ids)
        sentence_texts = [
            sentence_view.text.strip() for sentence_view in sentence_views
        ]
        sentence_groups = self.merger.merge(sentence_texts)

        chunks = []
        chunk_starts = []
        chunk_ends = []
        max_chunk_tokens = 0

        for subchunk_idx, sentence_indices in enumerate(sentence_groups):
            merged_sentence_texts = [sentence_texts[idx] for idx in sentence_indices]
            merged_text = " ".join(merged_sentence_texts).strip()
            merged_token_ids = self.model.tokenizer.encode(
                merged_text, add_special_tokens=False
            )
            if not merged_token_ids:
                continue

            first_sentence_view = sentence_views[sentence_indices[0]]
            last_sentence_view = sentence_views[sentence_indices[-1]]
            chunk_start = first_sentence_view.token_start
            chunk_end = last_sentence_view.token_end
            max_chunk_tokens = max(max_chunk_tokens, len(merged_token_ids))
            chunk_starts.append(chunk_start)
            chunk_ends.append(chunk_end)
            sentence_ids = [f"{filename}::sent_{idx}" for idx in sentence_indices]
            chunk_id = f"{filename}::{self.merger.name}::subchunk_{subchunk_idx}"
            chunks.append(
                CacheableChunk(
                    id=chunk_id,
                    text=merged_text,
                    parent_doc_id=filename,
                    chunk_size=self.cacheable_chunk_size,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    sentence_ids=sentence_ids,
                    sentence_texts=merged_sentence_texts,
                    prompt_token_count=self._prompt_visible_token_count(merged_text),
                    prompt_tokenizer_name=self.prompt_tokenizer_name,
                )
            )

        return chunks, chunk_starts, chunk_ends, max_chunk_tokens
