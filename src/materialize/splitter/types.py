"""Shared parsed-unit data types."""

from dataclasses import dataclass


@dataclass
class ParsedUnit:
    """One parser-produced text span aligned to source token offsets."""

    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
