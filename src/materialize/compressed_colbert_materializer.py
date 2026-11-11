"""Build a 2-bit subchunk store from one complete FP16 encoding pass."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch

import colbert_artifact
from encoder.colbert import default_colbert_repo_path
from materialize.colbert_materializer import (
    build_colbert_window_artifact,
    _validate_center_unit_against_db_manifest,
    add_region_specs_to_colbert_window_data,
    validate_colbert_subchunk_ids_against_db,
)
from materialize.colbert_residual import compress_colbert_window_artifact
from subchunk_store import load_subchunks


def build_compressed_colbert_subchunks(
    output_dir: str,
    db_dir: str,
    num_centroids: int | None = None,
    training_subchunks: int | None = None,
    model_name: str = "lightonai/LateOn",
    batch_size: int = 32,
    db_batch_size: int = 2048,
    window_token_budget: int = 180,
    sample_size: int | None = None,
    seed: int = 123,
    kmeans_niters: int = 4,
    batch_tokens: int = 4096,
    device: str = "cuda",
    use_gpu: bool = True,
    repo_path: str | None = None,
    validation_batch_size: int = 2048,
    verify_tensorization: bool = False,
    center_unit: str = "subchunk_only",
    disable_cpu_extension: bool = True,
    docs_dir: str | None = None,
):
    """Encode once to FP16, then train and compress from the stored vectors."""
    if training_subchunks is not None:
        raise ValueError(
            "training_subchunks changes global token sampling; use sample_size instead"
        )
    for name, value in (
        ("batch_size", batch_size),
        ("db_batch_size", db_batch_size),
        ("window_token_budget", window_token_budget),
        ("kmeans_niters", kmeans_niters),
        ("batch_tokens", batch_tokens),
        ("validation_batch_size", validation_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if num_centroids is not None and num_centroids <= 1:
        raise ValueError("num_centroids must be greater than one")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    manifest, reference = colbert_artifact.build_db_manifest_reference(
        db_dir=db_dir, artifact_dir=output
    )
    _validate_center_unit_against_db_manifest(manifest, center_unit)
    subchunks = load_subchunks(str(Path(db_dir).resolve()))
    if not subchunks:
        raise ValueError("cannot materialize an empty subchunk store")
    torch.manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.fp16-", dir=output.parent)
    )
    source = intermediate / "artifact"
    # Preserve a failed build/conversion for inspection. Remove our own FP16
    # intermediate only after the final compressed artifact passes validation.
    build_colbert_window_artifact(
        docs_dir=docs_dir or db_dir,
        output_dir=str(source),
        db_dir=db_dir,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        db_batch_size=db_batch_size,
        window_token_budget=window_token_budget,
        overwrite=False,
        repo_path=repo_path or default_colbert_repo_path(),
        disable_cpu_extension=disable_cpu_extension,
        verify_tensorization=verify_tensorization,
        center_unit=center_unit,
    )
    add_region_specs_to_colbert_window_data(
        source / "data", db_dir, window_token_budget
    )
    validate_colbert_subchunk_ids_against_db(
        str(source), db_dir, batch_size=validation_batch_size
    )
    compress_colbert_window_artifact(
        source,
        output,
        num_centroids=num_centroids,
        sample_size=sample_size,
        seed=seed,
        kmeans_niters=kmeans_niters,
        batch_tokens=batch_tokens,
        use_gpu=use_gpu,
    )
    validation = validate_colbert_subchunk_ids_against_db(
        str(output), db_dir, batch_size=validation_batch_size
    )
    summary = json.loads((output / "index.json").read_text())
    summary["validation"] = validation
    (output / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.rmtree(intermediate)
    return summary
