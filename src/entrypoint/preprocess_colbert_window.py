import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize.colbert_window import (  # noqa: E402
    add_region_specs_to_colbert_window_data,
    build_colbert_window_artifact,
    db_document_ids,
    default_colbert_repo_path,
    validate_colbert_window_artifact_against_db,
)


def main(
    docs_dir: str,
    output_dir: str,
    db_dir: str,
    model_name: str = "colbert-ir/colbertv2.0",
    batch_size: int = 32,
    window_token_budget: int = 180,
    overwrite: bool = False,
    repo_path: str | None = None,
    disable_cpu_extension: bool = True,
    validate_against_db: bool = True,
    validation_batch_size: int = 2048,
    prefix_title: bool = False,
    title_separator: str = "[SEP]",
    center_unit: str = "sentence",
    fixed_chunk_size: int | None = None,
):
    include_doc_ids = db_document_ids(db_dir=db_dir, batch_size=validation_batch_size)
    print(
        "Restricting ColBERT window artifact to "
        f"{len(include_doc_ids)} documents present in DB."
    )

    summary = build_colbert_window_artifact(
        docs_dir=docs_dir,
        output_dir=output_dir,
        db_dir=db_dir,
        model_name=model_name,
        device="cuda",
        batch_size=batch_size,
        window_token_budget=window_token_budget,
        overwrite=overwrite,
        repo_path=repo_path or default_colbert_repo_path(),
        disable_cpu_extension=disable_cpu_extension,
        verify_tensorization=False,
        prefix_title=prefix_title,
        title_separator=title_separator,
        center_unit=center_unit,
        fixed_chunk_size=fixed_chunk_size,
        include_doc_ids=include_doc_ids,
    )
    print(f"ColBERT window artifact materialized: {summary}")

    data_summary = add_region_specs_to_colbert_window_data(
        data_dir=Path(output_dir) / "data",
        db_dir=db_dir,
        region_token_budget=window_token_budget,
        overwrite=True,
    )
    printable = dict(data_summary)
    printable["id_to_row_count"] = len(printable.get("id_to_row", {}))
    printable["window_ids_by_row_count"] = len(printable.get("window_ids_by_row", []))
    printable["region_specs_by_chunk_count"] = len(
        printable.get("region_specs_by_chunk", {})
    )
    printable.pop("id_to_row", None)
    printable.pop("window_ids_by_row", None)
    printable.pop("region_specs_by_chunk", None)
    print(f"ColBERT window data materialized: {printable}")

    if validate_against_db:
        validation_summary = validate_colbert_window_artifact_against_db(
            artifact_dir=output_dir,
            db_dir=db_dir,
            batch_size=validation_batch_size,
        )
        print(f"ColBERT window artifact DB validation passed: {validation_summary}")


if __name__ == "__main__":
    fire.Fire(main)
