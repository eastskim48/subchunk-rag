import sys
import os
from pathlib import Path

import fire
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import LLMModel
from vectordb import ChromaDB
from materialize.materialize import DocumentPreprocessor
from materialize.db_manifest import build_db_manifest, write_db_build_manifest
from embedding_utils import BGE_M3_MODEL


class TokenizerOnlyModel:
    PASSAGE_PREFIX = LLMModel.PASSAGE_PREFIX

    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"LOADING TOKENIZER {model_name} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("TOKENIZER LOADED", flush=True)


def main(
    docs_dir: str,
    db_dir: str,
    cache_dir: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cacheable_chunk_size: int | None = 1024,
    retrievable_chunk_size: int | None = None,
    batch_size: int = 1,
    dummy_bos_count: int = 0,
    splitter: str = "fixed_size",
    merger: str | None = None,
    materialize_cache: bool = True,
    materialize_db: bool = True,
    materialize_compare_embeds: bool = False,
    compare_embed_dir: str | None = None,
    compare_embed_model: str = BGE_M3_MODEL,
    compare_embed_overwrite: bool = False,
    sentence_cache_token_format: str = "legacy",
    resume_from_cache: bool = False,
    materialize_doc_ids_file: str | None = None,
    deduplicate_documents_by_hash: bool = False,
    max_subchunk_tokens: int | None = None,
):
    model = (
        LLMModel(model_name) if materialize_cache else TokenizerOnlyModel(model_name)
    )
    preprocessor = DocumentPreprocessor(
        docs_dir=docs_dir,
        vectordb=ChromaDB(db_dir) if materialize_db else None,
        model=model,
        cache_dir=cache_dir,
        cacheable_chunk_size=cacheable_chunk_size,
        retrievable_chunk_size=retrievable_chunk_size,
        batch_size=batch_size,
        dummy_bos_count=dummy_bos_count,
        splitter=splitter,
        merger=merger,
        materialize_cache=materialize_cache,
        materialize_db=materialize_db,
        materialize_compare_embeds=materialize_compare_embeds,
        compare_embed_dir=compare_embed_dir,
        compare_embed_model=compare_embed_model,
        compare_embed_overwrite=compare_embed_overwrite,
        sentence_cache_token_format=sentence_cache_token_format,
        resume_from_cache=resume_from_cache,
        materialize_doc_ids_file=materialize_doc_ids_file,
        deduplicate_documents_by_hash=deduplicate_documents_by_hash,
        max_subchunk_tokens=max_subchunk_tokens,
    )

    preprocessor.process_documents()

    if materialize_db:
        manifest = build_db_manifest(
            splitter=splitter,
            merger=merger,
            cacheable_chunk_size=cacheable_chunk_size,
            retrievable_chunk_size=retrievable_chunk_size,
            max_subchunk_tokens=max_subchunk_tokens,
            tokenizer_name=model_name,
            dummy_bos_count=dummy_bos_count,
            sentence_cache_token_format=sentence_cache_token_format,
            deduplicate_documents_by_hash=deduplicate_documents_by_hash,
            embedding_backend=os.getenv(
                "CHROMA_EMBED_BACKEND", ChromaDB.DEFAULT_EMBED_BACKEND
            ),
        )
        manifest_path = write_db_build_manifest(db_dir, manifest)
        print(f"DB build manifest materialized: {manifest_path}")


if __name__ == "__main__":
    fire.Fire(main)
