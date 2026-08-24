"""Shared prompt-visible token accounting for selection compressors."""

import math
import os
from typing import List

from transformers import AutoTokenizer

from chunk import RetrievableChunk


class TokenBudgetMixin:
    """Resolve ratio/absolute budgets and cache token lengths consistently."""

    def _initialize_token_budget(self, require_budget: bool = True) -> None:
        # Grid configs use an empty string to explicitly clear an inherited
        # budget controller; treat it the same as an unset environment value.
        retain_token_ratio = (os.getenv("RETAIN_TOKEN_RATIO") or "").strip() or None
        final_token_budget = (os.getenv("FINAL_TOKEN_BUDGET") or "").strip() or None
        if retain_token_ratio is not None and final_token_budget is not None:
            raise ValueError(
                "exactly one of RETAIN_TOKEN_RATIO or FINAL_TOKEN_BUDGET may be "
                "set when a token budget is used"
            )
        if require_budget and retain_token_ratio is None and final_token_budget is None:
            raise ValueError(
                "exactly one of RETAIN_TOKEN_RATIO or FINAL_TOKEN_BUDGET must be set"
            )

        self.retain_token_ratio = (
            self._parse_retain_token_ratio(retain_token_ratio)
            if retain_token_ratio is not None
            else None
        )
        self.final_token_budget = (
            int(final_token_budget) if final_token_budget is not None else None
        )
        if self.final_token_budget is not None and self.final_token_budget <= 0:
            raise ValueError(
                f"FINAL_TOKEN_BUDGET must be positive, got {self.final_token_budget}"
            )

        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError(
                "MODEL_NAME must be set for final token-budget accounting; "
                "the final prompt budget must use the evaluated LLM tokenizer"
            )
        self.budget_tokenizer_name = model_name
        self.budget_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._token_len_cache: dict[str, int] = {}

    @staticmethod
    def _parse_retain_token_ratio(raw_value: str) -> float:
        value = raw_value.strip()
        if value.endswith("%"):
            value = value[:-1].strip()
            ratio = float(value) / 100.0
        else:
            ratio = float(value)
            if ratio > 1.0:
                ratio /= 100.0
        if ratio <= 0:
            raise ValueError(f"RETAIN_TOKEN_RATIO must be positive, got {raw_value!r}")
        return ratio

    def _retrieved_context_token_count(self, docs: List[RetrievableChunk]) -> int:
        seen_ids = set()
        unique_cacheables = []
        for doc in docs:
            for cacheable in getattr(doc, "cacheables", []) or []:
                text = getattr(cacheable, "text", None)
                if not text:
                    continue
                cacheable_id = getattr(cacheable, "id", None)
                if cacheable_id:
                    if cacheable_id in seen_ids:
                        continue
                    seen_ids.add(cacheable_id)
                unique_cacheables.append(cacheable)
        return sum(self._cacheable_token_lens(unique_cacheables))

    def _resolve_final_token_budget(self, docs: List[RetrievableChunk]) -> int | None:
        """Return the absolute prompt-token budget, or None when it is optional."""

        if self.retain_token_ratio is None:
            return self.final_token_budget
        retrieved_tokens = self._retrieved_context_token_count(docs)
        if retrieved_tokens <= 0:
            return 0
        return max(1, math.ceil(retrieved_tokens * self.retain_token_ratio))

    def _stored_prompt_token_count(self, cacheable) -> int | None:
        prompt_token_count = getattr(cacheable, "prompt_token_count", None)
        if prompt_token_count is None:
            return None
        prompt_tokenizer_name = getattr(cacheable, "prompt_tokenizer_name", None)
        if prompt_tokenizer_name != self.budget_tokenizer_name:
            return None
        return int(prompt_token_count)

    def _cacheable_token_len(self, cacheable) -> int:
        cacheable_id = getattr(cacheable, "id", None)
        if cacheable_id:
            cached = self._token_len_cache.get(cacheable_id)
            if cached is not None:
                return cached
        stored_token_count = self._stored_prompt_token_count(cacheable)
        if stored_token_count is not None:
            if cacheable_id:
                self._token_len_cache[cacheable_id] = stored_token_count
            return stored_token_count
        token_len = self._prompt_visible_token_count(cacheable)
        if cacheable_id:
            self._token_len_cache[cacheable_id] = token_len
        return token_len

    def _cacheable_token_lens(self, cacheables) -> list[int]:
        lengths: list[int | None] = []
        missing = []
        missing_positions = []
        for position, cacheable in enumerate(cacheables):
            cacheable_id = getattr(cacheable, "id", None)
            cached = self._token_len_cache.get(cacheable_id) if cacheable_id else None
            if cached is None:
                stored_token_count = self._stored_prompt_token_count(cacheable)
                if stored_token_count is not None:
                    if cacheable_id:
                        self._token_len_cache[cacheable_id] = stored_token_count
                    lengths.append(stored_token_count)
                    continue
                lengths.append(None)
                missing.append(cacheable)
                missing_positions.append(position)
            else:
                lengths.append(cached)

        if missing:
            token_lengths = self._prompt_visible_token_counts(missing)
            for cacheable, position, token_len in zip(
                missing, missing_positions, token_lengths
            ):
                lengths[position] = token_len
                cacheable_id = getattr(cacheable, "id", None)
                if cacheable_id:
                    self._token_len_cache[cacheable_id] = token_len

        return [int(length) for length in lengths if length is not None]

    def _format_budget_cacheable_text(self, cacheable) -> str:
        return f"{cacheable.text.strip()}\n\n"

    def _prompt_visible_token_count(self, cacheable) -> int:
        return self._prompt_visible_token_counts([cacheable])[0]

    def _prompt_visible_token_counts(self, cacheables) -> list[int]:
        texts = [
            self._format_budget_cacheable_text(cacheable) for cacheable in cacheables
        ]
        if not texts:
            return []
        encoded = self.budget_tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            verbose=False,
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    @staticmethod
    def _unique_cacheables(cacheables) -> list:
        unique = []
        seen_ids = set()
        for cacheable in cacheables:
            cacheable_id = getattr(cacheable, "id", None)
            if cacheable_id:
                if cacheable_id in seen_ids:
                    continue
                seen_ids.add(cacheable_id)
            unique.append(cacheable)
        return unique
