import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import LLMModel
from vectordb import ChromaDB
from materialize.colbert_window import (
    build_colbert_window_artifact,
    default_colbert_repo_path,
)
from materialize.materialize import DocumentPreprocessor
from embedding_utils import BGE_M3_MODEL


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
    sentence_resolver: str = "openai",
    openai_model: str = "gpt-4o-mini",
    fastcoref_model_name: str = "biu-nlp/f-coref",
    pn_mapping_dir: str | None = None,
    materialize_colbert_window: bool = False,
    colbert_window_dir: str | None = None,
    colbert_window_model: str = "colbert-ir/colbertv2.0",
    colbert_window_device: str | None = None,
    colbert_window_batch_size: int = 32,
    colbert_window_token_budget: int = 180,
    colbert_window_overwrite: bool = False,
    colbert_source_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    colbert_repo_path: str | None = None,
    colbert_disable_cpu_extension: bool = True,
    colbert_verify_tensorization: bool = True,
    colbert_window_center_unit: str = "sentence",
    colbert_window_fixed_chunk_size: int | None = None,
    colbert_window_prefix_title: bool = False,
    colbert_window_title_separator: str = "[SEP]",
):
    preprocessor = DocumentPreprocessor(
        docs_dir=docs_dir,
        vectordb=ChromaDB(db_dir),
        model=LLMModel(model_name),
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
        sentence_resolver=sentence_resolver,
        openai_model=openai_model,
        fastcoref_model_name=fastcoref_model_name,
        pn_mapping_dir=pn_mapping_dir,
    )

    preprocessor.process_documents()

    if materialize_colbert_window:
        output_dir = colbert_window_dir or str(Path(db_dir).parent / "colbert_window")
        repo_path = colbert_repo_path or default_colbert_repo_path()
        summary = build_colbert_window_artifact(
            docs_dir=docs_dir,
            output_dir=output_dir,
            source_tokenizer_name=colbert_source_tokenizer_name,
            model_name=colbert_window_model,
            device=colbert_window_device,
            batch_size=colbert_window_batch_size,
            window_token_budget=colbert_window_token_budget,
            overwrite=colbert_window_overwrite,
            repo_path=repo_path,
            disable_cpu_extension=colbert_disable_cpu_extension,
            verify_tensorization=colbert_verify_tensorization,
            center_unit=colbert_window_center_unit,
            fixed_chunk_size=colbert_window_fixed_chunk_size,
            prefix_title=colbert_window_prefix_title,
            title_separator=colbert_window_title_separator,
        )
        print(f"ColBERT window artifact materialized: {summary}")


if __name__ == "__main__":
    fire.Fire(main)
