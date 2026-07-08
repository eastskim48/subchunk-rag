import os
import inspect
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_SMALL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_M3_QUERY_PREFIX = ""


def default_query_prefix(model_name: str) -> str:
    if model_name == BGE_M3_MODEL:
        return BGE_M3_QUERY_PREFIX
    return BGE_SMALL_QUERY_PREFIX


class DenseTextEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 128,
        max_length: int | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length or (
            8192 if self.model_name == BGE_M3_MODEL else None
        )
        self.uses_flag_embedding = self.model_name == BGE_M3_MODEL

        if self.uses_flag_embedding:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise ImportError(
                    "BAAI/bge-m3 requires FlagEmbedding. Install dependencies from src/requirements.txt."
                ) from exc
            kwargs = {"use_fp16": self.device.startswith("cuda")}
            init_params = inspect.signature(BGEM3FlagModel.__init__).parameters
            if "devices" in init_params:
                kwargs["devices"] = self.device
            elif "device" in init_params:
                kwargs["device"] = self.device
            self.model = BGEM3FlagModel(self.model_name, **kwargs)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_dtype = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=model_dtype,
            ).to(self.device)
            self.model.eval()

    @property
    def embedding_dim(self) -> int:
        if self.uses_flag_embedding:
            return 1024
        return int(getattr(self.model.config, "hidden_size", 0))

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.uses_flag_embedding:
                encoded = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=self.max_length,
                )
                dense_vecs = (
                    encoded["dense_vecs"] if isinstance(encoded, dict) else encoded
                )
                batch_embeddings = torch.from_numpy(np.asarray(dense_vecs)).to(
                    torch.float32
                )
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )
            else:
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
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))
