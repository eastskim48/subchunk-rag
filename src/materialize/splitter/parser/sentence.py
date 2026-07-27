"""Sentence parser with conservative boundary repair and long-unit splitting."""

import re

from materialize.splitter.parser.base import UnitParser
from materialize.splitter.types import ParsedUnit


class SentenceParser(UnitParser):
    """Produce sentence units while preserving source character/token spans."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
        for sentence_text in sentence_texts:
            try:
                char_start, char_end = self.locate_span(
                    text, sentence_text, search_cursor
                )
            except ValueError:
                # Some corpora (notably TriviaQA paste dumps) contain repeated,
                # poorly formatted QA blobs that make sentence-to-source span
                # rematching ambiguous mid-document. In that case, keep the
                # remaining tail as a single sentence unit instead of aborting
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

        return units

    def split_long_units(
        self,
        units: list[ParsedUnit],
        token_ids: list[int],
        max_unit_tokens: int | None,
    ) -> list[ParsedUnit]:
        if max_unit_tokens is None:
            return units

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

    def _split_sentence_texts(self, text: str) -> list[str]:
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
