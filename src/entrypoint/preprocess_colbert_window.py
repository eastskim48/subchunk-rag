import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize.colbert_window import (  # noqa: E402
    build_colbert_window_artifact,
    default_colbert_repo_path,
    validate_colbert_window_artifact_against_db,
)


def main(
    docs_dir: str,
    output_dir: str,
    db_dir: str | None = None,
    source_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    model_name: str = "colbert-ir/colbertv2.0",
    device: str | None = None,
    batch_size: int = 32,
    window_token_budget: int = 180,
    overwrite: bool = False,
    repo_path: str | None = None,
    disable_cpu_extension: bool = True,
    verify_tensorization: bool = True,
    validate_against_db: bool = True,
    validation_batch_size: int = 2048,
    prefix_title: bool = False,
    title_separator: str = "[SEP]",
):
    summary = build_colbert_window_artifact(
        docs_dir=docs_dir,
        output_dir=output_dir,
        source_tokenizer_name=source_tokenizer_name,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        window_token_budget=window_token_budget,
        overwrite=overwrite,
        repo_path=repo_path or default_colbert_repo_path(),
        disable_cpu_extension=disable_cpu_extension,
        verify_tensorization=verify_tensorization,
        prefix_title=prefix_title,
        title_separator=title_separator,
    )
    print(f"ColBERT window artifact materialized: {summary}")

    if validate_against_db:
        if not db_dir:
            raise ValueError("db_dir is required when validate_against_db=True")
        validation_summary = validate_colbert_window_artifact_against_db(
            artifact_dir=output_dir,
            db_dir=db_dir,
            batch_size=validation_batch_size,
        )
        print(f"ColBERT window artifact DB validation passed: {validation_summary}")


if __name__ == "__main__":
    fire.Fire(main)
