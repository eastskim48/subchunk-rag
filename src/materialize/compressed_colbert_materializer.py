"""Build a 2-bit subchunk store from one complete FP16 encoding pass."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import colbert_artifact
from encoder.colbert import default_colbert_repo_path
from materialize.colbert_materializer import (
    ColBERTWindowEncoder,
    WindowSpec,
    _ColBERTArtifactWriter,
    build_colbert_window_artifact,
    _validate_center_unit_against_db_manifest,
    add_region_specs_to_colbert_window_data,
    validate_colbert_subchunk_ids_against_db,
)
from materialize.colbert_residual import (
    _default_num_centroids,
    _import_faiss,
    _make_faiss_index,
    _sha256_file,
    _train_codec_from_sample,
    compress_colbert_window_artifact,
    pack_2bit_values,
)
from materialize.db_cacheables import iter_unique_db_cacheables_by_document
from subchunk_store import load_subchunks


class _CompressedVectorSink:
    """Accept the existing writer's FP16 batches but persist only 2-bit records."""

    def __init__(self, handle, centroids, cutoffs, weights, use_gpu, batch_tokens):
        self.handle = handle
        self.centroids = centroids
        self.cutoffs = cutoffs
        self.weights = weights
        self.dim = centroids.shape[1]
        self.batch_bytes = batch_tokens * self.dim * 2
        self.pending = bytearray()
        self.digest = hashlib.sha256()
        self.source_digest = hashlib.sha256()
        self.index, self.resources = _make_faiss_index(
            _import_faiss(), centroids.astype(np.float32), use_gpu=use_gpu
        )

    @property
    def closed(self):
        return self.handle.closed

    def write(self, data):
        self.source_digest.update(data)
        self.pending.extend(data)
        while len(self.pending) >= self.batch_bytes:
            self._write_records(bytes(self.pending[: self.batch_bytes]))
            del self.pending[: self.batch_bytes]
        return len(data)

    def _write_records(self, data):
        batch = np.frombuffer(data, dtype=np.float16).reshape(-1, self.dim)
        batch = batch.astype(np.float32)
        _, codes = self.index.search(batch, 1)
        codes = codes[:, 0].astype(np.int32)
        residuals = batch - self.centroids[codes].astype(np.float32)
        quantized = np.searchsorted(self.cutoffs, residuals, side="left").astype(
            np.uint8
        )
        reconstructed = self.centroids[codes].astype(np.float32)
        reconstructed += self.weights[quantized]
        norms = np.linalg.norm(reconstructed, axis=1).astype(np.float32)
        np.maximum(norms, np.finfo(np.float32).eps, out=norms)
        records = np.empty(
            len(batch), dtype=colbert_artifact.colbert_2bit_token_record_dtype(self.dim)
        )
        records["code"] = codes
        records["residuals"] = pack_2bit_values(quantized)
        records["norm"] = norms
        payload = records.tobytes()
        self.handle.write(payload)
        self.digest.update(payload)

    def close(self):
        if self.closed:
            return
        try:
            if self.pending:
                self._write_records(bytes(self.pending))
                self.pending.clear()
        finally:
            self.handle.close()
            self.index = None
            self.resources = None


def _document_specs(encoder, db_dir, db_batch_size, window_token_budget, center_unit):
    for doc_id, cacheables in iter_unique_db_cacheables_by_document(
        db_dir, batch_size=db_batch_size
    ):
        if not cacheables:
            continue
        units = [item.text for item in cacheables]
        regions = encoder.build_centered_windows(units, window_token_budget)
        specs = (
            regions
            if center_unit == "subchunk"
            else [
                WindowSpec(text, 0, len(text), [i], [i], False)
                for i, text in enumerate(units)
            ]
        )
        yield doc_id, [item.id for item in cacheables], specs, regions


def _spec_batches(documents, batch_size):
    pending = []
    next_row = 0
    for _, ids, specs, _ in documents:
        for row, spec in zip(ids, specs, strict=True):
            if type(row) is not int or row != next_row:
                raise ValueError("DB subchunk order differs from vector row order")
            next_row += 1
            pending.append(spec)
            if len(pending) == batch_size:
                yield pending
                pending = []
    if pending:
        yield pending


def _count_tokens(encoder, batches, num_rows):
    counts = np.empty(num_rows, dtype=np.int64)
    row = 0
    for specs in tqdm(batches, desc="count stored ColBERT tokens"):
        values = encoder.document_span_token_counts(
            [s.text for s in specs], [(s.center_start, s.center_end) for s in specs]
        )
        if len(values) != len(specs) or row + len(values) > num_rows:
            raise ValueError("token-count pass returned an invalid row count")
        counts[row : row + len(values)] = values
        row += len(values)
    if row != num_rows or np.any(counts < 0):
        raise ValueError("token-count pass does not cover every subchunk row")
    return counts


def _training_vectors(encoder, batches, counts, sample_indices):
    """Keep global sampled tokens, preserving the original encoding batches."""
    order = np.argsort(sample_indices)
    sorted_indices = sample_indices[order]
    sample = np.empty((len(order), encoder.dim), dtype=np.float16)
    row = token = selected = 0
    for specs in tqdm(batches, desc="encode global codec sample"):
        batch_counts = counts[row : row + len(specs)]
        batch_end = token + int(batch_counts.sum())
        end = int(np.searchsorted(sorted_indices, batch_end, side="left"))
        if end > selected:
            vectors = encoder.encode_windows(specs)
            if len(vectors) != len(specs):
                raise ValueError("sample encoder returned an invalid row count")
            position = token
            for vector, count in zip(vectors, batch_counts, strict=True):
                if tuple(vector.shape) != (int(count), encoder.dim):
                    raise ValueError("sample vector lengths differ from counted tokens")
                stop = int(
                    np.searchsorted(sorted_indices, position + count, side="left")
                )
                if stop > selected:
                    positions = sorted_indices[selected:stop] - position
                    values = vector.contiguous().to(torch.float16).cpu().numpy()
                    sample[order[selected:stop]] = values[positions]
                    selected = stop
                position += int(count)
        row += len(specs)
        token = batch_end
    if row != len(counts) or selected != len(sample_indices):
        raise ValueError("global token sample was not completely encoded")
    return sample


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
