"""Abstract grouping interfaces with shared token-length accounting."""

from abc import ABC, abstractmethod

from materialize.splitter.types import ParsedUnit


def _env_token_budget(default: int = 128) -> int:
    import os

    return int(os.getenv("TOKEN_BUDGET", str(default)))


class UnitGrouper(ABC):
    """Group ordered parsed units by returning lists of unit indices."""

    name = "base"

    @abstractmethod
    def group(self, units: list[ParsedUnit]) -> list[list[int]]:
        raise NotImplementedError


class TokenBudgetGrouper(UnitGrouper):
    """Base grouper that precomputes tokenizer-visible unit lengths."""

    def __init__(self, tokenizer, token_budget: int | None = None):
        self.tokenizer = tokenizer
        self.token_budget = (
            _env_token_budget() if token_budget is None else token_budget
        )

    def _build_token_length_views(self, unit_texts: list[str]):
        first_token_lengths = [
            len(self.tokenizer.encode(text, add_special_tokens=False))
            for text in unit_texts
        ]
        continued_token_lengths = [
            len(self.tokenizer.encode(f" {text}", add_special_tokens=False))
            for text in unit_texts
        ]
        continued_prefix = [0]
        for value in continued_token_lengths:
            continued_prefix.append(continued_prefix[-1] + value)

        def chunk_token_len(start: int, end: int) -> int:
            if start >= end:
                return 0
            return first_token_lengths[start] + (
                continued_prefix[end] - continued_prefix[start + 1]
            )

        return chunk_token_len
