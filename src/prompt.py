"""Canonical prompt construction shared by preprocessing and inference."""

from typing import List


class PromptProcessor:
    """Build prompt-visible passage and query text in cache-compatible order."""

    SUPPORTED_FORMATS = {"raw_chunk_first", "chat_system_user"}

    def __init__(
        self,
        tokenizer,
        system_prompt: str,
        passage_prefix: str = "",
        prompt_format: str = "raw_chunk_first",
    ):
        if prompt_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported prompt_format={prompt_format!r}; "
                f"expected one of {sorted(self.SUPPORTED_FORMATS)}"
            )
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.passage_prefix = passage_prefix
        self.prompt_format = prompt_format

    def format_passage_chunk(self, passage: str) -> str:
        return f"{self.passage_prefix}{passage.strip()}\n\n"

    def format_passages(self, passages: List[str]) -> str:
        return "".join(
            self.format_passage_chunk(passage)
            for passage in passages
            if passage and passage.strip()
        )

    def build_query_prompt(self, query: str) -> str:
        if self.prompt_format == "chat_system_user":
            return self._build_chat_prompt(query=query, passages=[])
        return f"{self.system_prompt}\n\n" f"Question: {query.strip()}\n" "Answer:"

    def build_qa_prompt(self, query: str, passages: List[str]) -> str:
        if self.prompt_format == "chat_system_user":
            return self._build_chat_prompt(query=query, passages=passages)
        return f"{self.format_passages(passages)}{self.build_query_prompt(query)}"

    def build_cache_aligned_qa_prompt(self, query: str, passages: List[str]) -> str:
        return self.build_qa_prompt(query=query, passages=passages)

    def _build_chat_prompt(self, query: str, passages: List[str]) -> str:
        user_content = f"{self.format_passages(passages)}Question: {query.strip()}"
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            prompt = prompt[len(self.tokenizer.bos_token) :]
        return prompt

    def tokenize_full_prompts(self, prompts: List[str]):
        original_padding_side = self.tokenizer.padding_side
        original_truncation_side = self.tokenizer.truncation_side
        try:
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"
            return self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
            ).to("cuda")
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.truncation_side = original_truncation_side
