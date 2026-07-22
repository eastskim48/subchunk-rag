from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


class Chunk:
    def __init__(self, id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.text = text
        self.metadata = metadata or {}

    def _clone_common_kwargs(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": deepcopy(self.metadata),
        }


class CacheableChunk(Chunk):
    def __init__(
        self,
        id: str,
        text: str,
        parent_doc_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        sentence_ids: Optional[List[str]] = None,
        sentence_texts: Optional[List[str]] = None,
        prompt_token_count: Optional[int] = None,
        prompt_tokenizer_name: Optional[str] = None,
    ):
        self.id = id
        self.text = text
        self.parent_doc_id = parent_doc_id
        self.chunk_size = chunk_size
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.sentence_ids = list(sentence_ids or [])
        self.sentence_texts = list(sentence_texts or [])
        self.prompt_token_count = prompt_token_count
        self.prompt_tokenizer_name = prompt_tokenizer_name

    def clone(self) -> CacheableChunk:
        return CacheableChunk(
            id=self.id,
            text=self.text,
            parent_doc_id=self.parent_doc_id,
            chunk_size=self.chunk_size,
            chunk_start=self.chunk_start,
            chunk_end=self.chunk_end,
            sentence_ids=deepcopy(self.sentence_ids),
            sentence_texts=deepcopy(self.sentence_texts),
            prompt_token_count=self.prompt_token_count,
            prompt_tokenizer_name=self.prompt_tokenizer_name,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "parent_doc_id": self.parent_doc_id,
            "chunk_size": self.chunk_size,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "sentence_ids": list(self.sentence_ids),
            "sentence_texts": list(self.sentence_texts),
            "prompt_token_count": self.prompt_token_count,
            "prompt_tokenizer_name": self.prompt_tokenizer_name,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> CacheableChunk:
        return cls(
            id=payload["id"],
            text=payload["text"],
            parent_doc_id=payload.get("parent_doc_id"),
            chunk_size=payload.get("chunk_size"),
            chunk_start=payload.get("chunk_start"),
            chunk_end=payload.get("chunk_end"),
            sentence_ids=payload.get("sentence_ids"),
            sentence_texts=payload.get("sentence_texts"),
            prompt_token_count=payload.get("prompt_token_count"),
            prompt_tokenizer_name=payload.get("prompt_tokenizer_name"),
        )


class RetrievableChunk(Chunk):
    def __init__(
        self,
        id: str,
        text: str,
        cacheables: Optional[List[CacheableChunk]] = None,
        chunk_size: Optional[int] = None,
        token_count: Optional[int] = None,
        cache_unit: Optional[str] = None,
        token_budget: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id=id, text=text, metadata=metadata)
        self.cacheables = list(cacheables or [])
        self.chunk_size = chunk_size
        self.token_count = token_count
        self.cache_unit = cache_unit
        self.token_budget = token_budget

    def clone(self) -> RetrievableChunk:
        return RetrievableChunk(
            **self._clone_common_kwargs(),
            cacheables=[cacheable.clone() for cacheable in self.cacheables],
            chunk_size=self.chunk_size,
            token_count=self.token_count,
            cache_unit=self.cache_unit,
            token_budget=self.token_budget,
        )
