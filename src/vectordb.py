import fire
import json
import chromadb
import abc
import os
import time
import numpy as np

from typing import Dict, List

from chunk import RetrievableChunk, CacheableChunk
from embedding_utils import (
    BGE_M3_MODEL,
    DenseTextEmbedder,
    default_query_prefix,
    env_int,
)


class DenseEmbeddingFunction:
    MODEL_NAME = BGE_M3_MODEL

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 128,
        query_prefix: str | None = None,
        function_name: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or os.getenv("CHROMA_EMBED_DEVICE")
        self.batch_size = env_int("CHROMA_EMBED_BATCH_SIZE", batch_size)
        self.function_name = function_name or (
            "bge_m3" if model_name == BGE_M3_MODEL else None
        )
        self.query_prefix = (
            default_query_prefix(self.model_name)
            if query_prefix is None
            else query_prefix
        )
        self.embedder = DenseTextEmbedder(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.device = self.embedder.device

    def __call__(self, input):
        return self._embed_texts_batched(list(input))

    def embed_query(self, input):
        return self._embed_texts_batched(
            [f"{self.query_prefix}{text}" for text in input]
        )

    def _embed_texts_batched(self, texts: List[str]):
        if not texts:
            return []

        embeddings = self.embedder.embed_texts(texts).numpy()
        return [np.array(row, dtype=np.float32) for row in embeddings]

    def name(self) -> str:
        if self.model_name == BGE_M3_MODEL:
            return self.function_name or "bge_m3"
        sanitized = "".join(
            character if character.isalnum() else "_"
            for character in self.model_name.lower()
        ).strip("_")
        return self.function_name or f"dense_{sanitized}"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, object]):
        return DenseEmbeddingFunction(
            model_name=str(config.get("model_name", DenseEmbeddingFunction.MODEL_NAME)),
            device=str(config["device"]) if config.get("device") else None,
            batch_size=int(config.get("batch_size", 128)),
            query_prefix=(
                str(config["query_prefix"])
                if config.get("query_prefix") is not None
                else None
            ),
            function_name=(
                str(config["function_name"])
                if config.get("function_name") is not None
                else None
            ),
        )

    def get_config(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "query_prefix": self.query_prefix,
            "function_name": self.function_name,
        }

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def validate_config(config: Dict[str, object]) -> None:
        return


class MatKVDenseEmbeddingFunction(DenseEmbeddingFunction):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("function_name", "matkv_bge_m3")
        super().__init__(*args, **kwargs)


class VectorDB(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def find_top_k_docs(self, top_k, queries: List[str]):
        pass

    @abc.abstractmethod
    def store(self, chunks: List[RetrievableChunk]):
        pass


class ChromaDB(VectorDB):
    DEFAULT_EMBEDDING_MODEL = BGE_M3_MODEL
    DEFAULT_EMBED_BACKEND = "default"
    BGE_M3_EMBED_BACKEND = "bge_m3"
    LEGACY_EMBED_BACKENDS = {"matkv_bge_m3"}
    DEFAULT_COLLECTION_CONFIGURATION = {
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "ef_search": 200,
            "max_neighbors": 32,
            "resize_factor": 1.2,
            "sync_threshold": 1000,
        }
    }

    def __init__(self, db_dir: str):
        super().__init__()
        self.db = self._get_chroma_client(db_dir)
        self.include_documents = os.environ.get(
            "RETRIEVAL_INCLUDE_DOCUMENTS", "True"
        ).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        self.last_find_timings = {}

    @staticmethod
    def _get_chroma_client(dir: str):
        chroma_client = chromadb.PersistentClient(path=dir)
        embed_backend = (
            os.environ.get(
                "CHROMA_EMBED_BACKEND",
                ChromaDB.DEFAULT_EMBED_BACKEND,
            )
            .strip()
            .lower()
        )
        if embed_backend in {"default", "chroma_default"}:
            return chroma_client.get_or_create_collection(name="doc_collection")
        if (
            embed_backend != ChromaDB.BGE_M3_EMBED_BACKEND
            and embed_backend not in ChromaDB.LEGACY_EMBED_BACKENDS
        ):
            raise ValueError(
                f"unsupported CHROMA_EMBED_BACKEND={embed_backend!r}; "
                "expected 'default', 'bge_m3', 'chroma_default', "
                "or 'matkv_bge_m3' for legacy DBs"
            )
        function_name = (
            "matkv_bge_m3"
            if embed_backend in ChromaDB.LEGACY_EMBED_BACKENDS
            else ChromaDB.BGE_M3_EMBED_BACKEND
        )
        return chroma_client.get_or_create_collection(
            name="doc_collection",
            configuration=ChromaDB.DEFAULT_COLLECTION_CONFIGURATION,
            embedding_function=DenseEmbeddingFunction(
                model_name=ChromaDB.DEFAULT_EMBEDDING_MODEL,
                function_name=function_name,
            ),
        )

    @staticmethod
    def _serialize_cacheables(cacheables: List[CacheableChunk]) -> str:
        return json.dumps([cacheable.to_payload() for cacheable in cacheables])

    @staticmethod
    def _deserialize_cacheables(value) -> List[CacheableChunk]:
        if not value:
            return []
        if isinstance(value, str):
            value = json.loads(value)
        return [
            CacheableChunk.from_payload(item)
            for item in value
            if isinstance(item, dict)
        ]

    def find_top_k_docs(self, top_k: int, queries: List[str]):
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        if top_k == 0:
            self.last_find_timings = {
                "query_time": 0.0,
                "postprocess_time": 0.0,
                "cacheable_deserialize_time": 0.0,
            }
            return [[] for _ in queries]
        include = (
            ["metadatas", "documents"] if self.include_documents else ["metadatas"]
        )
        query_start = time.perf_counter()
        outputs = self.db.query(query_texts=queries, n_results=top_k, include=include)
        query_time = time.perf_counter() - query_start
        postprocess_start = time.perf_counter()
        cacheable_deserialize_time = 0.0
        batch_docs = []
        for i in range(len(queries)):
            ids = outputs["ids"][i]
            documents = outputs.get("documents")
            documents = documents[i] if documents is not None else [None] * len(ids)
            metadatas = outputs.get("metadatas", [[] for _ in range(len(queries))])[i]
            docs = []
            for doc_id, text, metadata in zip(ids, documents, metadatas):
                metadata = dict(metadata or {})
                deserialize_start = time.perf_counter()
                cacheables = self._deserialize_cacheables(
                    metadata.pop("cacheables_json", None)
                )
                cacheable_deserialize_time += time.perf_counter() - deserialize_start
                if text is None:
                    text = self._document_text_from_cacheables(cacheables)
                docs.append(
                    RetrievableChunk(
                        id=doc_id,
                        text=text,
                        cacheables=cacheables,
                        chunk_size=metadata.pop("chunk_size", None),
                        token_count=metadata.pop("token_count", None),
                        cache_unit=metadata.pop("cache_unit", None),
                        token_budget=metadata.pop("token_budget", None),
                        metadata=metadata,
                    )
                )
            batch_docs.append(docs)
        postprocess_time = time.perf_counter() - postprocess_start
        self.last_find_timings = {
            "query_time": query_time,
            "postprocess_time": postprocess_time,
            "cacheable_deserialize_time": cacheable_deserialize_time,
        }
        return batch_docs

    @staticmethod
    def _document_text_from_cacheables(cacheables: List[CacheableChunk]) -> str:
        return "\n\n".join(
            cacheable.text
            for cacheable in cacheables
            if getattr(cacheable, "text", None)
        )

    def store(self, chunks: List[RetrievableChunk]):
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["cacheables_json"] = self._serialize_cacheables(
                getattr(chunk, "cacheables", None) or []
            )
            if chunk.chunk_size is not None:
                metadata["chunk_size"] = chunk.chunk_size
            if chunk.token_count is not None:
                metadata["token_count"] = chunk.token_count
            if chunk.cache_unit is not None:
                metadata["cache_unit"] = chunk.cache_unit
            if chunk.token_budget is not None:
                metadata["token_budget"] = chunk.token_budget
            metadatas.append(metadata)
        self.db.upsert(
            documents=[chunk.text for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
            metadatas=metadatas,
        )
