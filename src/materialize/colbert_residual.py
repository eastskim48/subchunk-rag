"""Convert FP16 ColBERT window artifacts to 2-bit residual storage."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import colbert_artifact

RESIDUAL_STORAGE_FORMAT = colbert_artifact.RESIDUAL_STORAGE_FORMAT
RESIDUAL_NBITS = 2


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


_UNPACK_2BIT_LUT = np.asarray(
    [
        [
            ((byte >> shift) & 1) | (((byte >> (shift - 1)) & 1) << 1)
            for shift in (7, 5, 3, 1)
        ]
        for byte in range(256)
    ],
    dtype=np.uint8,
)


def _default_num_centroids(num_tokens: int) -> int:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    requested = int(2 ** np.floor(np.log2(16 * np.sqrt(num_tokens))))
    return min(requested, num_tokens)


def _import_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "2-bit ColBERT residual conversion requires FAISS, which is not "
            "installed. Install a FAISS build compatible with this environment "
            "before running the conversion entrypoint."
        ) from exc
    return faiss


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _normalize_rows_inplace(values: np.ndarray) -> np.ndarray:
    """Normalize a writable float32 row matrix without another row matrix."""

    if values.dtype != np.float32 or not values.flags.writeable:
        raise ValueError("in-place normalization requires writable float32 values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.maximum(norms, np.finfo(np.float32).eps, out=norms)
    np.divide(values, norms, out=values)
    return values


def pack_2bit_values(values: np.ndarray) -> np.ndarray:
    """Pack four 2-bit values per byte in official ResidualCodec bit order."""

    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f"2-bit values must be rank 2, got shape={values.shape}")
    if values.shape[1] % 4 != 0:
        raise ValueError(
            "2-bit value width must be divisible by four, "
            f"got width={values.shape[1]}"
        )
    if values.size and int(values.max()) > 3:
        raise ValueError("2-bit values must be in [0, 3]")
    bits = np.stack((values & 1, (values >> 1) & 1), axis=-1)
    return np.packbits(bits.reshape(values.shape[0], -1), axis=1)


def unpack_2bit_values(packed: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Unpack official ResidualCodec bit order into values in [0, 3]."""

    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError(f"packed residuals must be rank 2, got shape={packed.shape}")
    if embedding_dim <= 0 or embedding_dim % 4 != 0:
        raise ValueError(
            "embedding_dim must be a positive multiple of four, " f"got {embedding_dim}"
        )
    expected_width = embedding_dim // 4
    if packed.shape[1] != expected_width:
        raise ValueError(
            "packed residual width does not match embedding_dim: "
            f"packed={packed.shape[1]}, expected={expected_width}"
        )
    return np.take(_UNPACK_2BIT_LUT, packed, axis=0).reshape(
        packed.shape[0], embedding_dim
    )


def decompress_2bit_rows(
    codes: np.ndarray,
    packed_residuals: np.ndarray,
    centroids: np.ndarray,
    bucket_weights: np.ndarray,
    profile: dict[str, float | int] | None = None,
) -> torch.Tensor:
    """Reconstruct and normalize selected ColBERT token vectors on CPU."""

    if profile is not None:
        return _decompress_2bit_rows_profiled(
            codes=codes,
            packed_residuals=packed_residuals,
            centroids=centroids,
            bucket_weights=bucket_weights,
            profile=profile,
        )

    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 1:
        raise ValueError(f"centroid codes must be rank 1, got shape={codes.shape}")
    centroids = np.asarray(centroids, dtype=np.float32)
    bucket_weights = np.asarray(bucket_weights, dtype=np.float32)
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be rank 2, got shape={centroids.shape}")
    if bucket_weights.shape != (4,):
        raise ValueError(
            f"2-bit bucket weights must have shape (4,), got {bucket_weights.shape}"
        )
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= len(centroids)):
        raise ValueError("centroid code is outside the stored centroid table")
    quantized = unpack_2bit_values(packed_residuals, centroids.shape[1])
    reconstructed = centroids[codes]
    np.add(reconstructed, bucket_weights[quantized], out=reconstructed)
    return torch.from_numpy(_normalize_rows_inplace(reconstructed))


def _profile_add(profile: dict[str, float | int], key: str, value: float | int) -> None:
    profile[key] = profile.get(key, 0) + value


def _decompress_2bit_rows_profiled(
    *,
    codes: np.ndarray,
    packed_residuals: np.ndarray,
    centroids: np.ndarray,
    bucket_weights: np.ndarray,
    profile: dict[str, float | int],
) -> torch.Tensor:
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    codes = np.asarray(codes, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float32)
    bucket_weights = np.asarray(bucket_weights, dtype=np.float32)
    _profile_add(profile, "input_prepare_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    if codes.ndim != 1:
        raise ValueError(f"centroid codes must be rank 1, got shape={codes.shape}")
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be rank 2, got shape={centroids.shape}")
    if bucket_weights.shape != (4,):
        raise ValueError(
            f"2-bit bucket weights must have shape (4,), got {bucket_weights.shape}"
        )
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= len(centroids)):
        raise ValueError("centroid code is outside the stored centroid table")
    _profile_add(profile, "validation_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    quantized = unpack_2bit_values(packed_residuals, centroids.shape[1])
    _profile_add(profile, "unpack_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    reconstructed = centroids[codes]
    _profile_add(profile, "centroid_gather_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    residual_values = bucket_weights[quantized]
    _profile_add(profile, "residual_lookup_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    np.add(reconstructed, residual_values, out=reconstructed)
    del residual_values
    _profile_add(profile, "add_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    normalized = _normalize_rows_inplace(reconstructed)
    _profile_add(profile, "normalize_sec", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    result = torch.from_numpy(normalized)
    _profile_add(profile, "tensor_wrap_sec", time.perf_counter() - stage_start)

    _profile_add(profile, "decompression_calls", 1)
    _profile_add(profile, "token_vectors", int(codes.shape[0]))
    _profile_add(profile, "code_input_bytes", int(codes.shape[0] * 4))
    _profile_add(profile, "packed_residual_input_bytes", int(packed_residuals.nbytes))
    _profile_add(profile, "decompress_total_sec", time.perf_counter() - total_start)
    return result


def _make_faiss_index(faiss, centroids: np.ndarray, use_gpu: bool):
    index = faiss.IndexFlatIP(int(centroids.shape[1]))
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    if not use_gpu:
        return index, None
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError(
            "COLBERT_RESIDUAL_USE_GPU=True requires a GPU-enabled FAISS build"
        )
    resources = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(resources, 0, index), resources


def _train_codec(
    vectors: np.memmap,
    *,
    num_centroids: int,
    sample_size: int,
    seed: int,
    kmeans_niters: int,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faiss = _import_faiss()
    if use_gpu and not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError(
            "COLBERT_RESIDUAL_USE_GPU=True requires a GPU-enabled FAISS build"
        )
    num_tokens, embedding_dim = vectors.shape
    if not 1 < num_centroids <= num_tokens:
        raise ValueError(
            "num_centroids must satisfy 1 < num_centroids <= num_tokens, "
            f"got centroids={num_centroids}, tokens={num_tokens}"
        )
    if not num_centroids < sample_size <= num_tokens:
        raise ValueError(
            "sample_size must satisfy num_centroids < sample_size <= num_tokens, "
            f"got sample={sample_size}, centroids={num_centroids}, tokens={num_tokens}"
        )
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(num_tokens, size=sample_size, replace=False)
    sample = np.asarray(vectors[sample_indices], dtype=np.float32)
    heldout_size = min(max(1, int(0.05 * sample_size)), 50_000)
    if sample_size - heldout_size < num_centroids:
        raise ValueError(
            "training sample after the held-out split must contain at least one "
            "row per centroid"
        )
    permutation = rng.permutation(sample_size)
    heldout = np.ascontiguousarray(sample[permutation[:heldout_size]])
    training = np.ascontiguousarray(sample[permutation[heldout_size:]])
    kmeans = faiss.Kmeans(
        embedding_dim,
        num_centroids,
        niter=kmeans_niters,
        gpu=use_gpu,
        seed=seed,
    )
    kmeans.train(training)
    centroids = _normalize_rows(kmeans.centroids).astype(np.float16)
    assignment_index, assignment_resources = _make_faiss_index(
        faiss, centroids.astype(np.float32), use_gpu=use_gpu
    )
    _, heldout_codes = assignment_index.search(heldout, 1)
    del assignment_index
    del assignment_resources
    heldout_residuals = heldout - centroids[heldout_codes[:, 0]].astype(np.float32)
    residual_tensor = torch.from_numpy(heldout_residuals)
    cutoffs = torch.quantile(
        residual_tensor,
        torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
    ).numpy()
    weights = torch.quantile(
        residual_tensor,
        torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float32),
    ).numpy()
    return centroids, cutoffs.astype(np.float32), weights.astype(np.float32)


def _copy_sidecar(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing ColBERT artifact sidecar: {source}")
    shutil.copy2(source, destination)


def compress_colbert_window_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    num_centroids: int | None = None,
    sample_size: int | None = None,
    seed: int = 123,
    kmeans_niters: int = 4,
    batch_tokens: int = 4096,
    use_gpu: bool = True,
    storage_layout: str = "interleaved",
) -> dict[str, Any]:
    """Create a rollback-safe 2-bit derivative of a finalized FP16 artifact."""

    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    if source_path == output_path:
        raise ValueError("output_dir must differ from source_dir")
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing residual artifact: {output_path}"
        )
    if batch_tokens <= 0:
        raise ValueError(f"batch_tokens must be positive, got {batch_tokens}")
    if kmeans_niters <= 0:
        raise ValueError(f"kmeans_niters must be positive, got {kmeans_niters}")
    if storage_layout not in {"interleaved", "split"}:
        raise ValueError(
            "storage_layout must be 'interleaved' or 'split', "
            f"got {storage_layout!r}"
        )

    source_index_path = source_path / "index.json"
    source_data_path = source_path / "data"
    source_data_index_path = source_data_path / "index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_data_index = json.loads(source_data_index_path.read_text(encoding="utf-8"))
    if source_index.get("format") != colbert_artifact.ARTIFACT_FORMAT:
        raise ValueError(
            f"unsupported source artifact format: {source_index.get('format')}"
        )
    if source_data_index.get("format") != colbert_artifact.DATA_ARTIFACT_FORMAT:
        raise ValueError(
            "2-bit conversion requires a finalized FP16 ColBERT data artifact"
        )
    if source_data_index.get("subchunk_id_layout") != "vector_row":
        raise ValueError("source artifact requires integer-ID conversion")
    db_reference = source_index.get("db_manifest")
    if not isinstance(db_reference, dict):
        raise ValueError("source artifact requires a DB manifest reference")
    colbert_artifact.read_referenced_db_manifest(
        artifact_dir=source_path,
        reference=db_reference,
    )
    vectors_file = source_data_index.get("vectors_file")
    if not isinstance(vectors_file, str) or not vectors_file:
        raise ValueError("source FP16 artifact requires vectors_file")
    num_tokens = int(source_data_index["num_tokens"])
    embedding_dim = int(source_data_index["embedding_dim"])
    if embedding_dim <= 0 or embedding_dim % 4 != 0:
        raise ValueError(
            "2-bit residual storage requires embedding_dim divisible by four, "
            f"got {embedding_dim}"
        )
    source_vectors_path = source_data_path / vectors_file
    expected_vector_bytes = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    source_stat = source_vectors_path.stat()
    if source_stat.st_size != expected_vector_bytes:
        raise ValueError(
            "source vector file size does not match its data index: "
            f"actual={source_stat.st_size}, expected={expected_vector_bytes}"
        )
    vectors = np.memmap(
        source_vectors_path,
        dtype=np.float16,
        mode="r",
        shape=(num_tokens, embedding_dim),
    )
    resolved_centroids = (
        _default_num_centroids(num_tokens) if num_centroids is None else num_centroids
    )
    resolved_sample_size = (
        min(num_tokens, 100 * resolved_centroids)
        if sample_size is None
        else sample_size
    )
    centroids, bucket_cutoffs, bucket_weights = _train_codec(
        vectors,
        num_centroids=resolved_centroids,
        sample_size=resolved_sample_size,
        seed=seed,
        kmeans_niters=kmeans_niters,
        use_gpu=use_gpu,
    )
    faiss = _import_faiss()
    assignment_index, assignment_resources = _make_faiss_index(
        faiss, centroids.astype(np.float32), use_gpu=use_gpu
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.building-", dir=output_path.parent
        )
    )
    start_time = time.perf_counter()
    try:
        staging_data = staging_path / "data"
        staging_data.mkdir()
        token_records_file = "token_records.bin"
        codes_file = "codes.int32.bin"
        residuals_file = "residuals.2bit.bin"
        centroids_file = "centroids.fp16.bin"
        cutoffs_file = "bucket_cutoffs.float32.npy"
        weights_file = "bucket_weights.float32.npy"
        offsets_file = source_data_index["offsets_file"]
        for sidecar in (offsets_file,):
            _copy_sidecar(source_data_path / sidecar, staging_data / sidecar)

        centroids.tofile(staging_data / centroids_file)
        np.save(staging_data / cutoffs_file, bucket_cutoffs)
        np.save(staging_data / weights_file, bucket_weights)
        source_digest = hashlib.sha256()
        token_records_digest = hashlib.sha256()
        codes_digest = hashlib.sha256()
        residuals_digest = hashlib.sha256()
        record_dtype = colbert_artifact.colbert_2bit_token_record_dtype(embedding_dim)
        with contextlib.ExitStack() as stack:
            token_records_handle = None
            codes_handle = None
            residuals_handle = None
            if storage_layout == "interleaved":
                token_records_handle = stack.enter_context(
                    (staging_data / token_records_file).open("wb")
                )
            else:
                codes_handle = stack.enter_context(
                    (staging_data / codes_file).open("wb")
                )
                residuals_handle = stack.enter_context(
                    (staging_data / residuals_file).open("wb")
                )
            for start in tqdm(
                range(0, num_tokens, batch_tokens),
                desc="compress ColBERT token vectors",
            ):
                end = min(start + batch_tokens, num_tokens)
                source_bytes = vectors[start:end].tobytes(order="C")
                source_digest.update(source_bytes)
                batch = np.frombuffer(source_bytes, dtype=np.float16).reshape(
                    end - start, embedding_dim
                )
                batch_float = np.asarray(batch, dtype=np.float32)
                _, codes = assignment_index.search(batch_float, 1)
                codes = np.asarray(codes[:, 0], dtype=np.int32)
                residuals = batch_float - centroids[codes].astype(np.float32)
                quantized = np.searchsorted(
                    bucket_cutoffs, residuals, side="left"
                ).astype(np.uint8)
                packed = pack_2bit_values(quantized)
                if storage_layout == "interleaved":
                    reconstructed = centroids[codes].astype(np.float32)
                    np.add(
                        reconstructed,
                        bucket_weights[quantized],
                        out=reconstructed,
                    )
                    token_norms = np.linalg.norm(
                        reconstructed, axis=1, keepdims=False
                    ).astype(np.float32, copy=False)
                    np.maximum(token_norms, np.finfo(np.float32).eps, out=token_norms)
                    records = np.empty(end - start, dtype=record_dtype)
                    records["code"] = codes
                    records["residuals"] = packed
                    records["norm"] = token_norms
                    record_bytes = records.tobytes(order="C")
                    token_records_handle.write(record_bytes)
                    token_records_digest.update(record_bytes)
                else:
                    code_bytes = codes.tobytes(order="C")
                    residual_bytes = packed.tobytes(order="C")
                    codes_handle.write(code_bytes)
                    residuals_handle.write(residual_bytes)
                    codes_digest.update(code_bytes)
                    residuals_digest.update(residual_bytes)

        final_source_stat = source_vectors_path.stat()
        if (
            final_source_stat.st_size != source_stat.st_size
            or final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise RuntimeError("source FP16 vector file changed during conversion")

        data_index = dict(source_data_index)
        data_index.pop("vectors_file", None)
        for obsolete_key in (
            "token_records_file",
            "token_records_sha256",
            "token_record_layout",
            "codes_file",
            "codes_sha256",
            "residuals_file",
            "residuals_sha256",
        ):
            data_index.pop(obsolete_key, None)
        data_index.update(
            {
                "format": (
                    colbert_artifact.RESIDUAL_DATA_ARTIFACT_FORMAT
                    if storage_layout == "interleaved"
                    else colbert_artifact.LEGACY_RESIDUAL_DATA_ARTIFACT_FORMAT
                ),
                "embedding_storage": (
                    RESIDUAL_STORAGE_FORMAT
                    if storage_layout == "interleaved"
                    else colbert_artifact.LEGACY_RESIDUAL_STORAGE_FORMAT
                ),
                "nbits": RESIDUAL_NBITS,
                "num_centroids": resolved_centroids,
                "centroids_file": centroids_file,
                "centroids_sha256": _sha256_file(staging_data / centroids_file),
                "bucket_cutoffs_file": cutoffs_file,
                "bucket_cutoffs_sha256": _sha256_file(staging_data / cutoffs_file),
                "bucket_weights_file": weights_file,
                "bucket_weights_sha256": _sha256_file(staging_data / weights_file),
                "source_vectors_sha256": source_digest.hexdigest(),
                "compression_seed": seed,
                "compression_sample_size": resolved_sample_size,
                "kmeans_niters": kmeans_niters,
            }
        )
        if storage_layout == "interleaved":
            data_index.update(
                {
                    "token_records_file": token_records_file,
                    "token_records_sha256": token_records_digest.hexdigest(),
                    "token_record_layout": {
                        "byte_order": "little",
                        "record_size": record_dtype.itemsize,
                        "fields": {
                            "code": {"offset": 0, "dtype": "int32"},
                            "residuals": {
                                "offset": 4,
                                "dtype": "uint8",
                                "length": embedding_dim // 4,
                            },
                            "norm": {
                                "offset": 4 + embedding_dim // 4,
                                "dtype": "float32",
                            },
                        },
                    },
                }
            )
        else:
            data_index.update(
                {
                    "codes_file": codes_file,
                    "codes_sha256": codes_digest.hexdigest(),
                    "residuals_file": residuals_file,
                    "residuals_sha256": residuals_digest.hexdigest(),
                }
            )
        (staging_data / "index.json").write_text(
            json.dumps(data_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        source_db_manifest = (source_path / db_reference["path"]).resolve()
        top_index = dict(source_index)
        top_index["db_manifest"] = {
            "path": os.path.relpath(source_db_manifest, start=output_path),
            "sha256": db_reference["sha256"],
        }
        source_variant = source_index.get("artifact_variant", "colbert_window")
        top_index["artifact_variant"] = f"{source_variant}_residual_2bit"
        top_index["embedding_source"] = "data/index.json#embedding_storage"
        top_index["source_artifact"] = {
            "path": os.path.relpath(source_path, start=output_path),
            "index_sha256": _sha256_file(source_index_path),
            "data_index_sha256": _sha256_file(source_data_index_path),
            "vectors_sha256": source_digest.hexdigest(),
        }
        top_index["compression_time_sec"] = time.perf_counter() - start_time
        (staging_path / "index.json").write_text(
            json.dumps(top_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing residual artifact: {output_path}"
            )
        os.replace(staging_path, output_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise
    finally:
        del assignment_index
        del assignment_resources

    return {
        "source_dir": str(source_path),
        "output_dir": str(output_path),
        "num_tokens": num_tokens,
        "embedding_dim": embedding_dim,
        "num_centroids": resolved_centroids,
        "sample_size": resolved_sample_size,
        "nbits": RESIDUAL_NBITS,
        "storage_layout": storage_layout,
        "source_vector_bytes": expected_vector_bytes,
        "compressed_vector_bytes": (
            num_tokens
            * (
                colbert_artifact.colbert_2bit_token_record_dtype(embedding_dim).itemsize
                if storage_layout == "interleaved"
                else np.dtype(np.int32).itemsize + embedding_dim * RESIDUAL_NBITS // 8
            )
        ),
        "token_norm_bytes": (
            num_tokens * np.dtype(np.float32).itemsize
            if storage_layout == "interleaved"
            else 0
        ),
    }
