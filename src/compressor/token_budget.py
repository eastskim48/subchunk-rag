"""Shared prompt-visible token accounting for selection compressors."""

import os

from materialize.db_manifest import read_db_build_manifest


class TokenBudgetMixin:
    """Require a fixed context-token budget and cache token lengths."""

    def _initialize_token_budget(self, *, db_dir: str | None = None) -> None:
        final_token_budget = (os.getenv("FINAL_TOKEN_BUDGET") or "").strip()
        if not final_token_budget:
            raise ValueError("FINAL_TOKEN_BUDGET must be set")
        self.final_token_budget = int(final_token_budget)
        if self.final_token_budget <= 0:
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
