from typing import List


class PromptProcessor:
    def __init__(self, tokenizer, system_prompt: str, passage_prefix: str = ""):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.passage_prefix = passage_prefix

    def format_passage_chunk(self, passage: str) -> str:
        return f"{self.passage_prefix}{passage.strip()}\n\n"

    def format_passages(self, passages: List[str]) -> str:
        return "".join(
            self.format_passage_chunk(passage)
            for passage in passages
            if passage and passage.strip()
        )

    def build_query_prompt(self, query: str) -> str:
        return f"{self.system_prompt}\n\n" f"Question: {query.strip()}\n" "Answer:"

    def render_chat_prompt(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"System: {self.system_prompt}\n\n" f"User: {user_content}\n\n" "Assistant:"
        )

    def build_qa_prompt(self, query: str, passages: List[str]) -> str:
        return f"{self.format_passages(passages)}{self.build_query_prompt(query)}"

    def build_cache_aligned_qa_prompt(self, query: str, passages: List[str]) -> str:
        return self.build_qa_prompt(query=query, passages=passages)

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
