"""LLM summarization baseline for retrieved context."""

from typing import List
from chunk import RetrievableChunk
from dotenv import load_dotenv
import os
from openai import OpenAI
import math
import json

from compressor.base import Compressor


class Summarizer(Compressor):
    """Summarize each retrieved batch through an OpenAI-compatible endpoint."""

    def __init__(self):
        super().__init__()
        load_dotenv()
        print("Summarization enabled. Initializing OpenAI client...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env, but summarize=True was requested."
            )
        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(**client_kwargs)
        self.keep_ratio_min = 0.7
        self.keep_ratio_max = 0.9

    def _length_budget(self, document_text: str) -> tuple[int, int, int]:
        words = document_text.split()
        original_words = max(len(words), 1)
        min_words = max(20, math.ceil(original_words * self.keep_ratio_min))
        max_words = max(min_words, math.ceil(original_words * self.keep_ratio_max))
        return original_words, min_words, max_words

    def _build_batch_prompt(self, rchunks: List[RetrievableChunk], query: str) -> str:
        parts = []
        for idx, rchunk in enumerate(rchunks):
            original_words, min_words, max_words = self._length_budget(rchunk.text)
            parts.append(
                f"[DOC {idx}]\n"
                f"id: {rchunk.id}\n"
                f"original_length_words: {original_words}\n"
                f"target_summary_words: {min_words}-{max_words}\n"
                f"text:\n{rchunk.text}"
            )
        docs_block = "\n\n".join(parts)
        return (
            f"Summarize each document separately for answering question: {query}\n"
            "Keep each summary factual and grounded in the source.\n"
            "Do not add unsupported information.\n"
            "Do not compress too aggressively: preserve most of the useful evidence, "
            "including names, dates, entities, comparisons, and qualifiers that may affect the answer.\n"
            f"For each document, keep roughly {self.keep_ratio_min * 100} to {self.keep_ratio_max * 100} of the original length.\n"
            "Remove only clearly irrelevant detail. If unsure, keep the detail rather than dropping it.\n\n"
            "Return valid JSON only in this format:\n"
            '{"summaries": [{"index": 0, "summary": "..."}]}\n\n'
            f"Documents:\n{docs_block}"
        )

    def _parse_batch_response(
        self, content: str, retrievable_chunks: List[RetrievableChunk]
    ) -> List[RetrievableChunk]:
        try:
            parsed = json.loads(content)
            items = parsed.get("summaries", [])
            by_index = {
                int(item["index"]): str(item.get("summary", "")).strip()
                for item in items
                if "index" in item
            }
        except Exception:
            by_index = {}

        summarized_docs = []
        for idx, rchunk in enumerate(retrievable_chunks):
            summary = by_index.get(idx, rchunk.text)
            cloned = rchunk.clone()
            cloned.text = summary or rchunk.text
            summarized_docs.append(cloned)
        return summarized_docs

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize retrieved documents for question answering. "
                            "Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_batch_prompt(docs, query),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            summarized_batches.append(self._parse_batch_response(content, docs))
        return summarized_batches

    def compress(self, document_text: str, query: str) -> str:
        original_words, min_words, max_words = self._length_budget(document_text)
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize retrieved documents for question answering. "
                        "Keep the summary factual and grounded in the source. "
                        "Do not add unsupported information. "
                        "Do not compress too aggressively: preserve most of the useful evidence, "
                        "including names, dates, entities, comparisons, and qualifiers that may affect the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Document:\n{document_text}\n\n"
                        f"Original length: about {original_words} words.\n"
                        f"Write a concise summary that keeps roughly 50% to 70% of the original length "
                        f"(target: {min_words} to {max_words} words).\n"
                        "Remove only clearly irrelevant detail. If unsure, keep the detail rather than dropping it.\n"
                        "Return only the summary."
                    ),
                },
            ],
            temperature=0,
        )
        summary = response.choices[0].message.content
        return summary.strip() if summary else ""
