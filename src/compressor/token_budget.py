"""Shared prompt-visible token accounting for selection compressors."""

import math
import os
from typing import List

from materialize.db_manifest import read_db_build_manifest

from chunk import RetrievableChunk


class TokenBudgetMixin:
    """Resolve ratio/absolute budgets and cache token lengths consistently."""

    def _initialize_token_budget(
        self, require_budget: bool = True, *, db_dir: str | None = None
    ) -> None:
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
        if db_dir is None:
            raise ValueError(
                "DB_DIR or DATASET_PATH/DATA_SUBDIR is required for tokenizer validation"
            )
        manifest = read_db_build_manifest(db_dir)
        if manifest is None:
            raise FileNotFoundError(f"missing DB build manifest: {db_dir}")
        tokenizer_name = manifest.get("tokenizer_name")
        if tokenizer_name != model_name:
            raise ValueError(
                "DB tokenizer_name does not match MODEL_NAME: "
                f"db={tokenizer_name!r}, runtime={model_name!r}, db_dir={db_dir!r}"
            )
        self.budget_tokenizer_name = tokenizer_name
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
                if cacheable_id is not None:
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

    def _stored_prompt_token_count(self, cacheable) -> int:
        prompt_token_count = getattr(cacheable, "prompt_token_count", None)
        if prompt_token_count is None:
            raise ValueError(
                "missing prompt_token_count for subchunk "
                f"{getattr(cacheable, 'id', None)!r}"
            )
        return int(prompt_token_count)

    def _cacheable_token_len(self, cacheable) -> int:
        stored_token_count = self._stored_prompt_token_count(cacheable)
        cacheable_id = getattr(cacheable, "id", None)
        if cacheable_id is not None:
            cached = self._token_len_cache.get(cacheable_id)
            if cached is not None:
                return cached
            self._token_len_cache[cacheable_id] = stored_token_count
        return stored_token_count

    def _cacheable_token_lens(self, cacheables) -> list[int]:
        return [self._cacheable_token_len(cacheable) for cacheable in cacheables]

    def _format_budget_cacheable_text(self, cacheable) -> str:
        return f"{cacheable.text.strip()}\n\n"

    @staticmethod
    def _unique_cacheables(cacheables) -> list:
        unique = []
        seen_ids = set()
        for cacheable in cacheables:
            cacheable_id = getattr(cacheable, "id", None)
            if cacheable_id is not None:
                if cacheable_id in seen_ids:
                    continue
                seen_ids.add(cacheable_id)
            unique.append(cacheable)
        return unique
