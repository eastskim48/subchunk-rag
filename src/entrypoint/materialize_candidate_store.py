"""Materialize and validate the candidate store for an existing DB."""

import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoder.colbert import default_colbert_repo_path  # noqa: E402
from encoder.dense import BGE_M3_MODEL  # noqa: E402
from materialize.colbert_materializer import (  # noqa: E402
    add_region_specs_to_colbert_window_data,
    build_colbert_window_artifact,
    validate_colbert_candidate_ids_against_db,
)
from materialize.dense_materializer import (  # noqa: E402
    build_dense_embedding_artifact_from_db,
)

COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"


def main(
    output_dir: str,
    db_dir: str,
    docs_dir: str | None = None,
    backend: str = "colbert",
    model_name: str | None = None,
    batch_size: int | None = None,
    db_batch_size: int = 2048,
    window_token_budget: int = 180,
    overwrite: bool = False,
    repo_path: str | None = None,
    disable_cpu_extension: bool = True,
    validation_batch_size: int = 2048,
    center_unit: str = "subchunk",
    fixed_chunk_size: int | None = None,
):
    normalized_backend = backend.strip().lower()
    if normalized_backend == "dense":
        summary = build_dense_embedding_artifact_from_db(
            db_dir=db_dir,
            output_dir=output_dir,
            embedding_model=model_name or BGE_M3_MODEL,
            embedding_batch_size=128 if batch_size is None else batch_size,
            db_batch_size=db_batch_size,
            cache_unit=None,
            overwrite=overwrite,
        )
        print(f"Dense embedding artifact materialized: {summary}")
        return
    if normalized_backend != "colbert":
        raise ValueError(
            "candidate-store backend must be 'colbert' or 'dense', " f"got {backend!r}"
        )
    if not docs_dir:
        raise ValueError("docs_dir must be set for backend='colbert'")

    resolved_model_name = model_name or COLBERT_MODEL_NAME
    resolved_batch_size = 32 if batch_size is None else batch_size

    summary = build_colbert_window_artifact(
        docs_dir=docs_dir,
        output_dir=output_dir,
        db_dir=db_dir,
        model_name=resolved_model_name,
        device="cuda",
        batch_size=resolved_batch_size,
        db_batch_size=db_batch_size,
        window_token_budget=window_token_budget,
        overwrite=overwrite,
        repo_path=repo_path or default_colbert_repo_path(),
        disable_cpu_extension=disable_cpu_extension,
        verify_tensorization=False,
        center_unit=center_unit,
        fixed_chunk_size=fixed_chunk_size,
    )
    print(f"ColBERT window artifact materialized: {summary}")

    data_summary = add_region_specs_to_colbert_window_data(
        data_dir=Path(output_dir) / "data",
        db_dir=db_dir,
        region_token_budget=window_token_budget,
        overwrite=True,
    )
    print(f"ColBERT window data materialized: {data_summary}")

    validation_summary = validate_colbert_candidate_ids_against_db(
        artifact_dir=output_dir,
        db_dir=db_dir,
        batch_size=validation_batch_size,
    )
    print(f"ColBERT candidate ID validation passed: {validation_summary}")


if __name__ == "__main__":
    fire.Fire(main)
