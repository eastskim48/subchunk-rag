"""Sentence parser with conservative boundary repair and long-unit splitting."""

import re

from materialize.splitter.parser.base import UnitParser
from materialize.splitter.types import ParsedUnit


class SentenceParser(UnitParser):
    """Produce sentence units while preserving source character/token spans."""

    def __init__(self, tokenizer, long_unit_tokenizer=None):
        self.tokenizer = tokenizer
        self.long_unit_tokenizer = long_unit_tokenizer

    def parse(self, text: str, token_ids: list[int]) -> list[ParsedUnit]:
        sentence_texts = self._split_sentence_texts(text)
        if not sentence_texts:
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_token_ids = list(encoded["input_ids"])
        if full_token_ids != list(token_ids):
            token_ids = full_token_ids
        offsets = list(encoded["offset_mapping"])

        units = []
        search_cursor = 0
        token_cursor = 0
        source_spans = None
        for sentence_index, sentence_text in enumerate(sentence_texts):
            try:
                char_start, char_end = self.locate_span(
                    text, sentence_text, search_cursor
                )
            except ValueError:
                if source_spans is None:
                    source_spans = self._source_sentence_spans(text, sentence_texts)
                source_start, char_end = source_spans[sentence_index]
                char_start = search_cursor
                while char_start < len(text) and text[char_start].isspace():
                    char_start += 1
                if not char_start <= source_start < char_end <= len(text):
                    raise ValueError(
                        "sentence source offsets do not follow the previous span: "
                        f"sentence_index={sentence_index}, cursor={search_cursor}, "
                        f"source_span={(source_start, char_end)}"
                    )
                # Keep characters omitted by BlingFire in the original-text slice,
                # but retain its sentence end instead of consuming the document tail.
            search_cursor = char_end

            while (
                token_cursor < len(offsets) and offsets[token_cursor][1] <= char_start
            ):
                token_cursor += 1
            token_start = token_cursor
            while token_cursor < len(offsets) and offsets[token_cursor][0] < char_end:
                token_cursor += 1
            token_end = token_cursor

            units.append(
                ParsedUnit(
                    text=text[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                )
            )
            if char_end == len(text):
                break

        units = self._split_on_blank_lines(units, offsets)
        return self._split_on_urls(units, offsets)

    @staticmethod
    def _split_on_urls(
        units: list[ParsedUnit], offsets: list[tuple[int, int]]
    ) -> list[ParsedUnit]:
        split_units = []
        for unit in units:
            matches = list(re.finditer(r"\bhttps?://[^\s<>\"']+", unit.text, re.I))
            if not matches:
                split_units.append(unit)
                continue

            # Isolate source URL strings without deleting attached punctuation
            # or changing the original document's token positions.
            boundaries = [0]
            for match in matches:
                boundaries.extend([match.start(), match.end()])
            boundaries.append(len(unit.text))
            token_cursor = unit.token_start
            for left, right in zip(boundaries, boundaries[1:]):
                while left < right and unit.text[left].isspace():
                    left += 1
                while right > left and unit.text[right - 1].isspace():
                    right -= 1
                if left == right:
                    continue

                char_start = unit.char_start + left
                char_end = unit.char_start + right
                while (
                    token_cursor < unit.token_end
                    and offsets[token_cursor][1] <= char_start
                ):
                    token_cursor += 1
                token_start = token_cursor
                while (
                    token_cursor < unit.token_end
                    and offsets[token_cursor][0] < char_end
                ):
                    token_cursor += 1
                split_units.append(
                    ParsedUnit(
                        text=unit.text[left:right],
                        char_start=char_start,
                        char_end=char_end,
                        token_start=token_start,
                        token_end=token_cursor,
                    )
                )
        return split_units

    @staticmethod
    def _split_on_blank_lines(
        units: list[ParsedUnit], offsets: list[tuple[int, int]]
    ) -> list[ParsedUnit]:
        split_units = []
        for unit in units:
            if "\n\n" not in unit.text:
                split_units.append(unit)
                continue

            # Split recovered source text after abbreviation merging, so a
            # blank-line boundary cannot be merged back across paragraphs.
            part_start = 0
            token_cursor = unit.token_start
            for boundary in re.finditer(r"\n{2,}|\Z", unit.text):
                left, right = part_start, boundary.start()
                part_start = boundary.end()
                while left < right and unit.text[left].isspace():
                    left += 1
                while right > left and unit.text[right - 1].isspace():
                    right -= 1
                if left == right:
                    continue

                char_start = unit.char_start + left
                char_end = unit.char_start + right
                while (
                    token_cursor < unit.token_end
                    and offsets[token_cursor][1] <= char_start
                ):
                    token_cursor += 1
                token_start = token_cursor
                while (
                    token_cursor < unit.token_end
                    and offsets[token_cursor][0] < char_end
                ):
                    token_cursor += 1
                split_units.append(
                    ParsedUnit(
                        text=unit.text[left:right],
                        char_start=char_start,
                        char_end=char_end,
                        token_start=token_start,
                        token_end=token_cursor,
                    )
                )
        return split_units

    def _source_sentence_spans(
        self, text: str, sentence_texts: list[str]
    ) -> list[tuple[int, int]]:
        from blingfire import text_to_sentences_and_offsets

        stripped = text.strip()
        source_shift = len(text) - len(text.lstrip())
        split_text, offsets = text_to_sentences_and_offsets(stripped)
        parts = split_text.splitlines()
        if len(parts) != len(offsets):
            raise ValueError("BlingFire sentence text/offset counts differ")

        merged_texts = []
        merged_spans = []
        for part, (start, end) in zip(parts, offsets):
            part = part.strip()
            if not part:
                continue
            start += source_shift
            end += source_shift
            if merged_texts and self._should_merge_with_next(merged_texts[-1], part):
                merged_texts[-1] = f"{merged_texts[-1]} {part}".strip()
                merged_spans[-1] = (merged_spans[-1][0], end)
            else:
                merged_texts.append(part)
                merged_spans.append((start, end))

        if merged_texts != sentence_texts:
            raise ValueError("BlingFire sentence offsets do not match sentence text")
        return merged_spans

    def split_long_units(
        self,
        units: list[ParsedUnit],
        token_ids: list[int],
        max_unit_tokens: int | None,
        source_text: str | None = None,
    ) -> list[ParsedUnit]:
        if max_unit_tokens is None:
            return units
        if self.long_unit_tokenizer is not None:
            return self._split_long_units_with_tokenizer(
                units, token_ids, max_unit_tokens, source_text
            )

        split_units = []
        for unit in units:
            token_length = unit.token_end - unit.token_start
            if token_length <= max_unit_tokens:
                split_units.append(unit)
                continue

            part_start = unit.token_start
            while part_start < unit.token_end:
                part_end = min(part_start + max_unit_tokens, unit.token_end)
                if part_end <= part_start:
                    raise ValueError(
                        "failed to split long sentence into a non-empty subchunk: "
                        f"token_start={unit.token_start}, token_end={unit.token_end}, "
                        f"part_start={part_start}, part_end={part_end}"
                    )
                part_tokens = token_ids[part_start:part_end]
                part_text = self.tokenizer.decode(
                    part_tokens, skip_special_tokens=True
                ).strip()
                if part_text:
                    split_units.append(
                        ParsedUnit(
                            text=part_text,
                            char_start=unit.char_start,
                            char_end=unit.char_end,
                            token_start=part_start,
                            token_end=part_end,
                        )
                    )
                part_start = part_end
        return split_units

    def _split_long_units_with_tokenizer(
        self, units, token_ids, max_unit_tokens, source_text
    ):
        if max_unit_tokens <= 0:
            raise ValueError("max_unit_tokens must be positive")
        if not units:
            return []
        tokenizer = self.long_unit_tokenizer
        measured = tokenizer(
            [unit.text for unit in units],
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        if all(len(ids) <= max_unit_tokens for ids in measured["input_ids"]):
            return units
        if source_text is None:
            raise ValueError(
                "source_text is required for a separate long-unit tokenizer"
            )
        source = self.tokenizer(
            source_text, add_special_tokens=False, return_offsets_mapping=True
        )
        if list(source["input_ids"]) != list(token_ids):
            raise ValueError("source_text does not match the document token IDs")
        source_offsets = source["offset_mapping"]
        result = []
        for unit, ids, native_offsets in zip(
            units, measured["input_ids"], measured["offset_mapping"], strict=True
        ):
            if len(ids) <= max_unit_tokens:
                result.append(unit)
                continue
            if source_text[unit.char_start : unit.char_end] != unit.text:
                raise ValueError("long unit does not match its source character span")
            start = 0
            token_cursor = unit.token_start
            while start < len(unit.text):
                while start < len(unit.text) and unit.text[start].isspace():
                    start += 1
                if start == len(unit.text):
                    break
                remaining = unit.text[start:]
                if start:
                    encoded = tokenizer(
                        remaining,
                        add_special_tokens=False,
                        truncation=False,
                        return_offsets_mapping=True,
                        verbose=False,
                    )
                    ids, native_offsets = (
                        encoded["input_ids"],
                        encoded["offset_mapping"],
                    )
                if len(ids) <= max_unit_tokens:
                    end = len(remaining)
                else:
                    end = min(
                        native_offsets[max_unit_tokens - 1][1],
                        native_offsets[max_unit_tokens][0],
                    )
                piece = remaining[:end].rstrip()
                # Cut only at complete source characters. Retokenization at the
                # new text boundary can change byte-pair merges and token count.
                while (
                    piece
                    and len(
                        tokenizer(piece, add_special_tokens=False, truncation=False)[
                            "input_ids"
                        ]
                    )
                    > max_unit_tokens
                ):
                    end = max(
                        (a for a, _ in native_offsets if 0 < a < len(piece)),
                        default=0,
                    )
                    piece = remaining[:end].rstrip()
                if not piece:
                    raise ValueError(
                        "cannot split a complete source character within budget"
                    )
                char_start = unit.char_start + start
                char_end = char_start + len(piece)
                while (
                    token_cursor < unit.token_end
                    and source_offsets[token_cursor][1] <= char_start
                ):
                    token_cursor += 1
                token_start = token_cursor
                while (
                    token_cursor < unit.token_end
                    and source_offsets[token_cursor][0] < char_end
                ):
                    token_cursor += 1
                result.append(
                    ParsedUnit(
                        text=piece,
                        char_start=char_start,
                        char_end=char_end,
                        token_start=token_start,
                        token_end=token_cursor,
                    )
                )
                start += end
        return result

    def _split_sentence_texts(self, text: str) -> list[str]:
        try:
            from blingfire import text_to_sentences
        except ImportError as exc:
            raise ImportError(
                "splitter='sentence' requires blingfire to be installed"
            ) from exc

        sentence_texts = [
            part.strip()
            for part in text_to_sentences(text.strip()).splitlines()
            if part.strip()
        ]
        if not sentence_texts and text.strip():
            sentence_texts = [text.strip()]
        return self._merge_conservative_boundaries(sentence_texts)

    @staticmethod
    def _should_merge_with_next(prev_unit: str, next_unit: str) -> bool:
        prev = prev_unit.strip()
        nxt = next_unit.strip()
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

    def _merge_conservative_boundaries(self, units: list[str]) -> list[str]:
        if not units:
            return units

        merged = []
        cursor = 0
        while cursor < len(units):
            current = units[cursor]
            while cursor + 1 < len(units) and self._should_merge_with_next(
                current,
                units[cursor + 1],
            ):
                current = f"{current} {units[cursor + 1]}".strip()
                cursor += 1
            merged.append(current)
            cursor += 1
        return merged

    @staticmethod
    def locate_span(text: str, unit_text: str, search_cursor: int) -> tuple[int, int]:
        text_cursor = search_cursor
        while text_cursor < len(text) and text[text_cursor].isspace():
            text_cursor += 1
        if text.startswith(unit_text, text_cursor):
            return text_cursor, text_cursor + len(unit_text)

        normalized_unit = " ".join(unit_text.split())
        unit_cursor = 0
        char_start = text_cursor

        while unit_cursor < len(normalized_unit):
            if text_cursor >= len(text):
                raise ValueError(
                    f"failed to locate sentence span in source document: {unit_text!r}"
                )

            unit_ch = normalized_unit[unit_cursor]
            text_ch = text[text_cursor]

            if unit_ch.isspace():
                if not text_ch.isspace():
                    raise ValueError(
                        f"failed to locate sentence span in source document: {unit_text!r}"
                    )
                while (
                    unit_cursor < len(normalized_unit)
                    and normalized_unit[unit_cursor].isspace()
                ):
                    unit_cursor += 1
                while text_cursor < len(text) and text[text_cursor].isspace():
                    text_cursor += 1
                continue

            if text_ch.isspace():
                raise ValueError(
                    f"failed to locate sentence span in source document: {unit_text!r}"
                )

            if text_ch != unit_ch:
                raise ValueError(
                    f"failed to locate sentence span in source document: {unit_text!r}"
                )

            unit_cursor += 1
            text_cursor += 1

        return char_start, text_cursor
