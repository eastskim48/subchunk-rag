"""CLI entrypoint for building chunks, caches, and the retrieval database."""

import sys
import os
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import LLMModel
from vectordb import ChromaDB
from materialize.materialize import DocumentPreprocessor
from materialize.db_manifest import build_db_manifest, write_db_build_manifest


def main(
    docs_dir: str,
    db_dir: str,
    cache_dir: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cacheable_chunk_size: int | None = None,
    retrievable_chunk_size: int | None = None,
    batch_size: int = 1,
    db_batch_size: int = 256,
    chroma_embed_device: str = "cpu",
    chroma_embed_batch_size: int = 32,
    dummy_bos_count: int = 0,
    materialize_cache: bool = True,
    materialize_db: bool = True,
    resume_from_cache: bool = False,
    deduplicate_documents_by_hash: bool = False,
    max_subchunk_tokens: int | None = None,
):
    preprocessor = DocumentPreprocessor(
        docs_dir=docs_dir,
        vectordb=(
            ChromaDB.for_build(
                db_dir,
                embedding_device=chroma_embed_device,
                embedding_batch_size=chroma_embed_batch_size,
            )
            if materialize_db
            else None
        ),
        model_name=model_name,
        cache_dir=cache_dir,
        cacheable_chunk_size=cacheable_chunk_size,
        retrievable_chunk_size=retrievable_chunk_size,
        batch_size=batch_size,
        db_batch_size=db_batch_size,
        dummy_bos_count=dummy_bos_count,
        materialize_cache=materialize_cache,
        materialize_db=materialize_db,
        resume_from_cache=resume_from_cache,
        deduplicate_documents_by_hash=deduplicate_documents_by_hash,
        max_subchunk_tokens=max_subchunk_tokens,
    )

    preprocessor.process_documents()

    if materialize_db:
        manifest = build_db_manifest(
            splitter="sentence",
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            max_subchunk_tokens=max_subchunk_tokens,
            tokenizer_name=model_name,
            deduplicate_documents_by_hash=deduplicate_documents_by_hash,
            embedding_backend=os.getenv(
                "CHROMA_EMBED_BACKEND", ChromaDB.DEFAULT_EMBED_BACKEND
            ),
            db_batch_size=db_batch_size,
            embedding_device=chroma_embed_device,
            embedding_batch_size=chroma_embed_batch_size,
        )
        manifest_path = write_db_build_manifest(db_dir, manifest)
        print(f"DB build manifest materialized: {manifest_path}")


if __name__ == "__main__":
    fire.Fire(main)
