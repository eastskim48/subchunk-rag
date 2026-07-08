import os
from typing import Dict, List

import torch

from chunk import CacheableChunk
from embedding_utils import BGE_M3_MODEL, DenseTextEmbedder, default_query_prefix


class CompareEmbeddingWriter:
    QUERY_PREFIX = default_query_prefix(BGE_M3_MODEL)
    OUTPUT_FILENAME = "compare_embed.pt"

    def __init__(
        self,
        output_dir: str,
        embedding_model: str = BGE_M3_MODEL,
        embedding_batch_size: int = 128,
        cache_unit: str = "sentence",
        overwrite: bool = False,
    ):
        self.output_dir = output_dir
        self.embedding_model_name = embedding_model
        self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_batch_size = embedding_batch_size
        self.cache_unit = cache_unit
        self.overwrite = overwrite

        if self.cache_unit not in {"token", "sentence"}:
            raise ValueError(f"unsupported cache_unit: {self.cache_unit}")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")

        self.query_prefix = default_query_prefix(self.embedding_model_name)
        self.embedding_backend = DenseTextEmbedder(
            model_name=self.embedding_model_name,
            device=self.embedding_device,
            batch_size=self.embedding_batch_size,
        )

        self.written_docs = 0
        self.skipped_docs = 0
        self.written_chunks = 0
        self.disabled = False
        self.pending_docs: List[Dict[str, object]] = []

        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self._output_path()) and not self.overwrite:
            self.disabled = True

    def _output_path(self) -> str:
        return os.path.join(self.output_dir, self.OUTPUT_FILENAME)

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        return self.embedding_backend.embed_texts(texts).to(self.embedding_device)

    def _embed_texts_batched(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            hidden_size = self.embedding_backend.embedding_dim
            return torch.empty((0, hidden_size), dtype=torch.float16)

        batches = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch_embeddings = self._embed_texts(
                texts[start : start + self.embedding_batch_size]
            )
            batches.append(batch_embeddings.detach().cpu())
        return (
            torch.cat(batches, dim=0)
            if batches
            else torch.empty((0, 0), dtype=torch.float16)
        )

    def write_document(self, doc_id: str, cacheables: List[CacheableChunk]):
        if self.disabled:
            return

        chunk_ids = []
        chunk_texts = []
        for cacheable in cacheables:
            if not cacheable.text:
                continue
            chunk_ids.append(cacheable.id)
            chunk_texts.append(cacheable.text)

        if not chunk_texts:
            return

        embeddings = self._embed_texts_batched(chunk_texts).to(torch.float16)
        self.pending_docs.append(
            {
                "doc_id": doc_id,
                "chunk_ids": chunk_ids,
                "embeddings": embeddings,
            }
        )
        self.written_docs += 1
        self.written_chunks += len(chunk_texts)

    def finalize(self):
        if self.disabled:
            return

        doc_index = {}
        all_embeddings = []
        row_cursor = 0
        for payload in self.pending_docs:
            embeddings = payload["embeddings"]
            row_count = int(embeddings.shape[0])
            all_embeddings.append(embeddings)
            doc_index[payload["doc_id"]] = {
                "row_start": row_cursor,
                "row_end": row_cursor + row_count,
                "chunk_ids": list(payload["chunk_ids"]),
            }
            row_cursor += row_count

        embeddings = (
            torch.cat(all_embeddings, dim=0)
            if all_embeddings
            else torch.empty((0, 0), dtype=torch.float16)
        )
        payload = {
            "format": "compare_embed_single_v1",
            "embedding_model": self.embedding_model_name,
            "embedding_query_prefix": self.query_prefix,
            "segment_unit": self.cache_unit,
            "embedding_dtype": "float16",
            "doc_index": doc_index,
            "embeddings": embeddings,
        }
        torch.save(payload, self._output_path())
