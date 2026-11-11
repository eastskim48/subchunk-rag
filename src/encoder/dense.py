"""BGE-small text encoding for retrieval."""

import os
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"
BGE_SMALL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def default_query_prefix(model_name: str) -> str:
    if model_name == BGE_SMALL_MODEL:
        return BGE_SMALL_QUERY_PREFIX
    return ""


class DenseTextEmbedder:
    """Load BGE-small and return normalized CLS text embeddings."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 128,
        max_length: int | None = None,
    ):
        if model_name != BGE_SMALL_MODEL:
            raise ValueError(
                f"unsupported retrieval embedding model: {model_name!r}; "
                f"expected {BGE_SMALL_MODEL!r}"
            )
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length or None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_dtype = (
            torch.float32 if not self.device.startswith("cuda") else torch.float16
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 0))

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = self._embed_texts_transformers(batch)
            batches.append(batch_embeddings.detach().cpu())
        return torch.cat(batches, dim=0)

    def _embed_texts_transformers(self, texts: List[str]) -> torch.Tensor:
        encode_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            encode_kwargs["max_length"] = self.max_length
        encoded = self.tokenizer(texts, **encode_kwargs).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)

        token_embeddings = outputs.last_hidden_state
        pooled = token_embeddings[:, 0]
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))
