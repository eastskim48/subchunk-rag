"""Build contextualized ColBERT candidate stores and region sidecar metadata."""

from __future__ import annotations

import json
import sqlite3
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import colbert_artifact
from colbert_metadata import (
    CACHEABLE_ROWS_FILE,
    ColBERTMetadataReader,
    ColBERTMetadataWriter,
    REGION_PAYLOADS_FILE,
    write_split_metadata_from_sqlite,
)
from encoder import colbert as colbert_encoder
from materialize.db_cacheables import (
    iter_db_cacheable_groups,
    iter_db_cacheables,
    iter_unique_db_cacheables_by_document,
)


@dataclass
class WindowSpec:
    """Text window plus the token span belonging to its center candidate."""

    text: str
    center_start: int
    center_end: int
    selected_indices: list[int]
    addition_order: list[int]
    truncated_center: bool


def _build_fixed_chunk_window_spec(
    *,
    center_unit: str,
    source_tokenizer,
    visible_token_overhead: int,
    source_token_ids: list[int],
    chunk_start: int,
    chunk_end: int,
    center_text: str,
    center_index: int,
    window_token_budget: int,
) -> WindowSpec:
    """Build one fixed center, optionally with source-token context."""

    if center_unit != "fixed_chunk_window":
        return WindowSpec(
            text=center_text,
            center_start=0,
            center_end=len(center_text),
            selected_indices=[center_index],
            addition_order=[center_index],
            truncated_center=False,
        )

    center_len = max(0, chunk_end - chunk_start)
    context_budget = max(0, window_token_budget - center_len - visible_token_overhead)
    left_budget = context_budget // 2
    right_budget = context_budget - left_budget
    left_start = max(0, chunk_start - left_budget)
    right_end = min(len(source_token_ids), chunk_end + right_budget)
    left_tokens = source_token_ids[left_start:chunk_start]
    right_tokens = source_token_ids[chunk_end:right_end]

    parts: list[str] = []
    cursor = 0
    if left_tokens:
        left_text = source_tokenizer.decode(
            left_tokens, skip_special_tokens=True
        ).strip()
        if left_text:
            parts.append(left_text)
            cursor += len(left_text)
            parts.append(" ")
            cursor += 1
    center_start = cursor
    parts.append(center_text)
    cursor += len(center_text)
    center_end = cursor
    if right_tokens:
        right_text = source_tokenizer.decode(
            right_tokens, skip_special_tokens=True
        ).strip()
        if right_text:
            parts.append(" ")
            parts.append(right_text)

    return WindowSpec(
        text="".join(parts),
        center_start=center_start,
        center_end=center_end,
        selected_indices=[center_index],
        addition_order=[center_index],
        truncated_center=False,
    )


class ColBERTWindowEncoder(colbert_encoder.ColBERTEncoder):
    """Construct centered windows and retain only each center unit's vectors."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int = 0,
        doc_maxlen: int | None = None,
        query_maxlen: int | None = None,
        attend_to_mask_tokens: bool | None = None,
        mask_punctuation: bool | None = None,
        repo_path: str | None = None,
        disable_cpu_extension: bool = True,
        verify_tensorization: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
            attend_to_mask_tokens=attend_to_mask_tokens,
            mask_punctuation=mask_punctuation,
            repo_path=repo_path,
            disable_cpu_extension=disable_cpu_extension,
            verify_tensorization=verify_tensorization,
        )
        self.max_length = int(max_length or self.doc_maxlen)
        if self.max_length > self.doc_maxlen:
            raise ValueError(
                "COLBERT_WINDOW_TOKEN_BUDGET cannot exceed the official ColBERT doc_maxlen: "
                f"budget={self.max_length}, doc_maxlen={self.doc_maxlen}"
            )

    @staticmethod
    def _next_window_candidate(
        state: dict[str, Any], sentence_count: int
    ) -> int | None:
        left = state["left"]
        right = state["right"]
        take_left = state["take_left"]

        if take_left and left >= 0:
            candidate_idx = left
            state["left"] = left - 1
        elif (not take_left) and right < sentence_count:
            candidate_idx = right
            state["right"] = right + 1
        elif left >= 0:
            candidate_idx = left
            state["left"] = left - 1
        elif right < sentence_count:
            candidate_idx = right
            state["right"] = right + 1
        else:
            return None

        state["take_left"] = not take_left
        return candidate_idx

    def build_centered_windows(
        self, sentences: list[str], window_token_budget: int
    ) -> list[WindowSpec]:
        if not sentences:
            return []

        if window_token_budget > self.doc_maxlen:
            raise ValueError(
                "ColBERT window token budget cannot exceed official doc_maxlen: "
                f"budget={window_token_budget}, doc_maxlen={self.doc_maxlen}"
            )

        sentence_token_counts = self.token_counts_without_specials(sentences)
        specs: list[WindowSpec | None] = [None] * len(sentences)
        states: list[dict[str, Any] | None] = []

        for center_idx, center_text in enumerate(sentences):
            center_token_count = (
                sentence_token_counts[center_idx] + self.doc_token_overhead
            )
            if center_token_count >= window_token_budget:
                specs[center_idx] = WindowSpec(
                    text=center_text,
                    center_start=0,
                    center_end=len(center_text),
                    selected_indices=[center_idx],
                    addition_order=[center_idx],
                    truncated_center=True,
                )
                states.append(None)
                continue

            states.append(
                {
                    "selected": {center_idx},
                    "addition_order": [center_idx],
                    "left": center_idx - 1,
                    "right": center_idx + 1,
                    "token_count": center_token_count,
                    "take_left": True,
                    "active": True,
                }
            )

        while any(state is not None and state["active"] for state in states):
            for center_idx, state in enumerate(states):
                if state is None or not state["active"]:
                    continue
                candidate_idx = self._next_window_candidate(state, len(sentences))
                if candidate_idx is None:
                    state["active"] = False
                    continue
                token_count = (
                    int(state["token_count"]) + sentence_token_counts[candidate_idx]
                )
                if token_count > window_token_budget:
                    state["active"] = False
                    continue
                state["selected"].add(candidate_idx)
                state["addition_order"].append(candidate_idx)
                state["token_count"] = token_count

        for center_idx, state in enumerate(states):
            if specs[center_idx] is not None:
                continue
            if state is None:
                raise RuntimeError(f"missing window state for center_idx={center_idx}")
            ordered = sorted(state["selected"])
            parts: list[str] = []
            center_start = 0
            cursor = 0
            for idx in ordered:
                if parts:
                    parts.append(" ")
                    cursor += 1
                if idx == center_idx:
                    center_start = cursor
                sentence = sentences[idx]
                parts.append(sentence)
                cursor += len(sentence)
            center_text = sentences[center_idx]
            specs[center_idx] = WindowSpec(
                text="".join(parts),
                center_start=center_start,
                center_end=center_start + len(center_text),
                selected_indices=ordered,
                addition_order=list(state["addition_order"]),
                truncated_center=False,
            )

        return [spec for spec in specs if spec is not None]

    def encode_windows(
        self, specs: list[WindowSpec], show_progress: bool = False
    ) -> list[torch.Tensor]:
        return self.encode_document_spans(
            texts=[spec.text for spec in specs],
            center_spans=[
                (int(spec.center_start), int(spec.center_end)) for spec in specs
            ],
            show_progress=show_progress,
        )


class _ColBERTArtifactWriter:
    """Batch window encoding and persist the compact candidate index."""

    def __init__(
        self,
        encoder: ColBERTWindowEncoder,
        vectors_path: Path,
        metadata_path: Path,
        batch_size: int,
    ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.num_docs = 0
        self.offsets: list[int] = []
        self.total_cacheables = 0
        self.total_center_tokens = 0
        self.truncated_centers = 0
        self.embedding_dim: int | None = None
        self.pending_docs: list[dict[str, Any]] = []
        self.pending_window_count = 0
        self.vector_handle = vectors_path.open("wb")
        self.metadata_writer = ColBERTMetadataWriter(metadata_path)

    def add_document(
        self,
        doc_id: str,
        cacheable_ids: list[str],
        specs: list[WindowSpec],
    ) -> None:
        self.pending_docs.append(
            {
                "doc_id": doc_id,
                "cacheable_ids": cacheable_ids,
                "specs": specs,
            }
        )
        self.pending_window_count += len(specs)
        if self.pending_window_count >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending_docs:
            return

        flat_specs = [
            spec for pending_doc in self.pending_docs for spec in pending_doc["specs"]
        ]
        flat_vectors = self.encoder.encode_windows(flat_specs) if flat_specs else []
        cursor = 0
        metadata_records = []

        for pending_doc in self.pending_docs:
            specs = pending_doc["specs"]
            vectors = flat_vectors[cursor : cursor + len(specs)]
            cursor += len(specs)
            doc_token_counts = [int(vector.shape[0]) for vector in vectors]
            if vectors and self.embedding_dim is None:
                self.embedding_dim = int(vectors[0].shape[1])

            cacheable_ids = pending_doc["cacheable_ids"]
            for cacheable_id, vector, spec in zip(cacheable_ids, vectors, specs):
                row_index = self.total_cacheables
                self.offsets.append(self.total_center_tokens)
                expected_dim = self.embedding_dim or self.encoder.dim
                if int(vector.shape[1]) != expected_dim:
                    raise ValueError(
                        f"embedding dim mismatch for {cacheable_id}: "
                        f"{vector.shape[1]} != {expected_dim}"
                    )
                vector = vector.contiguous().to(torch.float16).cpu()
                self.vector_handle.write(vector.numpy().tobytes(order="C"))
                self.total_center_tokens += int(vector.shape[0])
                window_ids = [
                    str(cacheable_ids[idx])
                    for idx in spec.selected_indices
                    if isinstance(idx, int) and 0 <= idx < len(cacheable_ids)
                ]
                metadata_records.append(
                    [
                        str(cacheable_id),
                        row_index,
                        window_ids,
                    ]
                )
                self.total_cacheables += 1
            self.truncated_centers += sum(1 for spec in specs if spec.truncated_center)
            self.num_docs += 1

        self.metadata_writer.add_cacheables(metadata_records)

        self.pending_docs = []
        self.pending_window_count = 0

    def finalize(self) -> None:
        self.flush()
        self.vector_handle.close()
        self.metadata_writer.close()

    def close(self) -> None:
        if not self.vector_handle.closed:
            self.vector_handle.close()
        self.metadata_writer.close()


def _validate_center_unit_against_db_manifest(
    db_manifest: dict[str, Any],
    center_unit: str,
    fixed_chunk_size: int | None,
) -> None:
    """Reject artifact candidate units that differ from the persisted DB units."""

    splitter = db_manifest.get("splitter")
    if splitter == "sentence":
        allowed = {"subchunk", "subchunk_only"}
    elif splitter in {"fixed_size", "fixed_subchunk"}:
        allowed = {"fixed_chunk", "fixed_chunk_window"}
    elif splitter == "semantic":
        raise ValueError(
            "DB splitter='semantic' has no supported ColBERT center_unit; "
            "materialization must not reinterpret semantic DB cacheables as sentences"
        )
    else:
        raise ValueError(f"unsupported DB manifest splitter: {splitter!r}")

    if center_unit not in allowed:
        raise ValueError(
            "ColBERT center_unit is incompatible with the DB splitter: "
            f"splitter={splitter!r}, center_unit={center_unit!r}, "
            f"allowed={sorted(allowed)}"
        )

    if splitter in {"fixed_size", "fixed_subchunk"}:
        db_chunk_size = db_manifest.get("cacheable_chunk_size")
        if (
            isinstance(db_chunk_size, bool)
            or not isinstance(db_chunk_size, int)
            or db_chunk_size <= 0
        ):
            raise ValueError(
                "fixed-size DB manifest requires a positive cacheable_chunk_size, "
                f"got {db_chunk_size!r}"
            )
        if fixed_chunk_size != db_chunk_size:
            raise ValueError(
                "fixed_chunk_size must match the DB cacheable_chunk_size: "
                f"fixed_chunk_size={fixed_chunk_size!r}, "
                f"cacheable_chunk_size={db_chunk_size}"
            )


def build_colbert_window_artifact(
    docs_dir: str,
    output_dir: str,
    db_dir: str,
    model_name: str = "colbert-ir/colbertv2.0",
    device: str | None = None,
    batch_size: int = 32,
    db_batch_size: int = 2048,
    window_token_budget: int = 0,
    overwrite: bool = False,
    repo_path: str | None = None,
    disable_cpu_extension: bool = True,
    verify_tensorization: bool = True,
    mask_punctuation: bool | None = None,
    center_unit: str = "subchunk",
    fixed_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Encode source windows and write the contextualized candidate store."""

    if center_unit not in {
        "subchunk",
        "subchunk_only",
        "fixed_chunk",
        "fixed_chunk_window",
    }:
        raise ValueError(
            "center_unit must be one of "
            "{'subchunk', 'subchunk_only', 'fixed_chunk', 'fixed_chunk_window'}, "
            f"got {center_unit!r}"
        )
    if center_unit in {"fixed_chunk", "fixed_chunk_window"} and (
        fixed_chunk_size is None or fixed_chunk_size <= 0
    ):
        raise ValueError(
            "fixed_chunk_size must be a positive integer when center_unit uses fixed chunks"
        )

    docs_path = Path(docs_dir)
    artifact_dir = Path(output_dir)
    data_dir = artifact_dir / "data"
    db_manifest, db_manifest_reference = colbert_artifact.build_db_manifest_reference(
        db_dir=db_dir, artifact_dir=artifact_dir
    )
    _validate_center_unit_against_db_manifest(
        db_manifest=db_manifest,
        center_unit=center_unit,
        fixed_chunk_size=fixed_chunk_size,
    )
    repo_path = repo_path or colbert_encoder.default_colbert_repo_path()
    if overwrite and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifact_dir / "index.json"
    data_index_path = data_dir / "index.json"
    metadata_path = data_dir / ".build_metadata.sqlite3"
    if (
        index_path.exists()
        and data_index_path.exists()
        and (data_dir / CACHEABLE_ROWS_FILE).exists()
        and (data_dir / REGION_PAYLOADS_FILE).exists()
        and not overwrite
    ):
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("format") != colbert_artifact.ARTIFACT_FORMAT:
            raise ValueError(
                "existing ColBERT artifact was built by an unsupported/non-official implementation; "
                "set COLBERT_WINDOW_OVERWRITE=True to rebuild it"
            )
        existing_reference = existing.get("db_manifest")
        if not isinstance(existing_reference, dict):
            raise ValueError("existing ColBERT v2 artifact is missing db_manifest")
        colbert_artifact.read_referenced_db_manifest(
            artifact_dir=artifact_dir, reference=existing_reference
        )
        if existing_reference.get("sha256") != db_manifest_reference["sha256"]:
            raise ValueError(
                "existing ColBERT artifact does not match the requested DB manifest"
            )
        data_existing = json.loads(data_index_path.read_text(encoding="utf-8"))
        if data_existing.get("format") != colbert_artifact.DATA_ARTIFACT_FORMAT:
            raise ValueError(
                "existing ColBERT data has unsupported format: "
                f"{data_existing.get('format')}"
            )
        return existing

    start_time = time.perf_counter()
    source_tokenizer = None
    visible_token_overhead = None
    fixed_body_chunk_size = None
    if center_unit in {"fixed_chunk", "fixed_chunk_window"}:
        source_tokenizer_name = db_manifest.get("tokenizer_name")
        if not isinstance(source_tokenizer_name, str) or not source_tokenizer_name:
            raise ValueError(
                "DB build manifest tokenizer_name must be a non-empty string"
            )

        source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_name)
        visible_token_overhead = len(
            source_tokenizer.encode("", add_special_tokens=False)
        ) + len(source_tokenizer.encode("\n\n", add_special_tokens=False))
        fixed_body_chunk_size = int(fixed_chunk_size) - visible_token_overhead
        if fixed_body_chunk_size <= 0:
            raise ValueError(
                "fixed_chunk_size must be larger than prompt-visible token overhead: "
                f"fixed_chunk_size={fixed_chunk_size}, overhead={visible_token_overhead}"
            )
    encoder = ColBERTWindowEncoder(
        model_name=model_name,
        repo_path=repo_path,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
        max_length=window_token_budget,
        mask_punctuation=mask_punctuation,
        disable_cpu_extension=disable_cpu_extension,
        verify_tensorization=verify_tensorization,
    )
    effective_token_budget = encoder.max_length
    content_token_budget = effective_token_budget

    vectors_file = "vectors.fp16.bin"
    offsets_file = "offsets.npy"
    skipped_existing = 0
    failed_docs: list[dict[str, str]] = []
    writer = _ColBERTArtifactWriter(
        encoder=encoder,
        vectors_path=data_dir / vectors_file,
        metadata_path=metadata_path,
        batch_size=batch_size,
    )
    num_input_docs = 0
    try:
        for doc_id, cacheables in tqdm(
            iter_unique_db_cacheables_by_document(
                db_dir=db_dir,
                batch_size=db_batch_size,
            ),
            desc="build colbert window artifact",
        ):
            num_input_docs += 1
            if not cacheables:
                continue
            try:
                source_text = None
                if center_unit == "fixed_chunk_window":
                    source_text = (docs_path / doc_id).read_text(encoding="utf-8")

                units = [cacheable.text for cacheable in cacheables]
                cacheable_ids = [str(cacheable.id) for cacheable in cacheables]
                if center_unit in {"subchunk", "subchunk_only"}:
                    if center_unit == "subchunk_only":
                        specs = [
                            WindowSpec(
                                text=unit,
                                center_start=0,
                                center_end=len(unit),
                                selected_indices=[idx],
                                addition_order=[idx],
                                truncated_center=False,
                            )
                            for idx, unit in enumerate(units)
                        ]
                    else:
                        specs = encoder.build_centered_windows(
                            sentences=units,
                            window_token_budget=content_token_budget,
                        )
                else:
                    if (
                        fixed_body_chunk_size is None
                        or source_tokenizer is None
                        or visible_token_overhead is None
                    ):
                        raise RuntimeError("fixed_body_chunk_size was not initialized")
                    source_token_ids = (
                        source_tokenizer.encode(source_text, add_special_tokens=False)
                        if source_text is not None
                        else []
                    )
                    specs = []
                    for center_index, cacheable in enumerate(cacheables):
                        chunk_start = cacheable.chunk_start
                        chunk_end = cacheable.chunk_end
                        if (
                            isinstance(chunk_start, bool)
                            or not isinstance(chunk_start, int)
                            or isinstance(chunk_end, bool)
                            or not isinstance(chunk_end, int)
                            or chunk_start < 0
                            or chunk_end <= chunk_start
                        ):
                            raise ValueError(
                                "fixed-size DB cacheable requires a valid source token "
                                f"span: id={cacheable.id!r}, start={chunk_start!r}, "
                                f"end={chunk_end!r}"
                            )
                        if center_unit == "fixed_chunk_window" and chunk_end > len(
                            source_token_ids
                        ):
                            raise ValueError(
                                "fixed-size DB cacheable exceeds the source document: "
                                f"id={cacheable.id!r}, end={chunk_end}, "
                                f"source_tokens={len(source_token_ids)}"
                            )
                        specs.append(
                            _build_fixed_chunk_window_spec(
                                center_unit=center_unit,
                                source_tokenizer=source_tokenizer,
                                visible_token_overhead=visible_token_overhead,
                                source_token_ids=source_token_ids,
                                chunk_start=chunk_start,
                                chunk_end=chunk_end,
                                center_text=cacheable.text,
                                center_index=center_index,
                                window_token_budget=content_token_budget,
                            )
                        )
            except Exception as exc:
                failed_docs.append({"doc_id": doc_id, "error": repr(exc)})
                continue

            writer.add_document(
                doc_id=doc_id,
                cacheable_ids=cacheable_ids,
                specs=specs,
            )
        writer.finalize()
    except BaseException:
        writer.close()
        raise

    final_offsets = np.asarray(
        writer.offsets + [writer.total_center_tokens], dtype=np.int64
    )
    np.save(data_dir / offsets_file, final_offsets)
    embedding_dim = int(writer.embedding_dim or encoder.dim)

    summary = {
        "format": colbert_artifact.ARTIFACT_FORMAT,
        "encoder_impl": "official_colbert_checkpoint",
        "data_dir": "data",
        "docs_dir": str(docs_path),
        "db_manifest": db_manifest_reference,
        "model_name": model_name,
        "checkpoint_name": model_name,
        "repo_path": repo_path,
        "device": encoder.device,
        "batch_size": batch_size,
        "window_token_budget": effective_token_budget,
        "official_doc_maxlen": encoder.doc_maxlen,
        "official_query_maxlen": encoder.query_maxlen,
        "mask_punctuation": bool(encoder.checkpoint.colbert_config.mask_punctuation),
        "embedding_dim": embedding_dim,
        "window_scope": "parent_document",
        "center_unit": center_unit,
        "fixed_chunk_size": fixed_chunk_size,
        "fixed_body_chunk_size": fixed_body_chunk_size,
        "num_docs": writer.num_docs,
        "num_input_docs": num_input_docs,
        "num_cacheables": writer.total_cacheables,
        "num_center_tokens": writer.total_center_tokens,
        "avg_center_tokens_per_cacheable": (
            writer.total_center_tokens / writer.total_cacheables
            if writer.total_cacheables
            else 0.0
        ),
        "truncated_centers": writer.truncated_centers,
        "skipped_existing": skipped_existing,
        "failed_docs": failed_docs,
        "build_time_sec": time.perf_counter() - start_time,
    }
    index_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    data_summary = {
        "format": "colbert_window_data_build_v1",
        "source_format": summary["format"],
        "embedding_dim": embedding_dim,
        "num_cacheables": writer.total_cacheables,
        "num_tokens": writer.total_center_tokens,
        "vectors_file": vectors_file,
        "offsets_file": offsets_file,
        "metadata_file": metadata_path.name,
        "region_token_budget": None,
        "region_spec_chunk_count": 0,
        "region_spec_count": 0,
        "build_time_sec": summary["build_time_sec"],
    }
    data_index_path.write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def validate_colbert_candidate_ids_against_db(
    artifact_dir: str,
    db_dir: str,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Require the materialized candidate IDs to match the DB exactly."""

    artifact = colbert_artifact.ColBERTWindowArtifact(artifact_dir)
    artifact_cacheable_count = len(artifact.data.id_to_row)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=".candidate_validation_",
        suffix=".sqlite3",
        dir=artifact.data.data_dir,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    connection = sqlite3.connect(temporary_path)
    duplicate_db_ids = 0
    total_db_occurrences = 0
    try:
        connection.execute("pragma journal_mode=off")
        connection.execute("pragma synchronous=off")
        connection.execute("create table db_ids (cacheable_id text primary key)")
        pending_ids = []
        for cacheable in iter_db_cacheables(db_dir=db_dir, batch_size=batch_size):
            total_db_occurrences += 1
            pending_ids.append((str(cacheable.id),))
            if len(pending_ids) >= batch_size:
                before = connection.total_changes
                connection.executemany(
                    "insert or ignore into db_ids (cacheable_id) values (?)",
                    pending_ids,
                )
                duplicate_db_ids += len(pending_ids) - (
                    connection.total_changes - before
                )
                pending_ids.clear()
        if pending_ids:
            before = connection.total_changes
            connection.executemany(
                "insert or ignore into db_ids (cacheable_id) values (?)",
                pending_ids,
            )
            duplicate_db_ids += len(pending_ids) - (connection.total_changes - before)
        connection.commit()
        connection.execute("create table artifact_ids (cacheable_id text primary key)")
        pending_artifact_ids = []
        for cacheable_id in artifact.data.id_to_row:
            pending_artifact_ids.append((str(cacheable_id),))
            if len(pending_artifact_ids) >= batch_size:
                connection.executemany(
                    "insert into artifact_ids (cacheable_id) values (?)",
                    pending_artifact_ids,
                )
                pending_artifact_ids.clear()
        if pending_artifact_ids:
            connection.executemany(
                "insert into artifact_ids (cacheable_id) values (?)",
                pending_artifact_ids,
            )
        connection.commit()
        db_cacheable_count = int(
            connection.execute("select count(*) from db_ids").fetchone()[0]
        )
        missing_in_artifact_count = int(
            connection.execute(
                "select count(*) from ("
                "select cacheable_id from db_ids "
                "except select cacheable_id from artifact_ids)"
            ).fetchone()[0]
        )
        extra_in_artifact_count = int(
            connection.execute(
                "select count(*) from ("
                "select cacheable_id from artifact_ids "
                "except select cacheable_id from db_ids)"
            ).fetchone()[0]
        )
        summary = {
            "db_dir": str(db_dir),
            "artifact_dir": str(artifact_dir),
            "db_cacheable_occurrences": total_db_occurrences,
            "db_cacheable_count": db_cacheable_count,
            "artifact_cacheable_count": artifact_cacheable_count,
            "duplicate_db_ids": duplicate_db_ids,
            "missing_in_artifact_count": missing_in_artifact_count,
            "extra_in_artifact_count": extra_in_artifact_count,
        }
        if missing_in_artifact_count or extra_in_artifact_count:
            missing_examples = [
                str(row[0])
                for row in connection.execute(
                    "select cacheable_id from db_ids "
                    "except select cacheable_id from artifact_ids limit 10"
                )
            ]
            extra_examples = [
                str(row[0])
                for row in connection.execute(
                    "select cacheable_id from artifact_ids "
                    "except select cacheable_id from db_ids limit 10"
                )
            ]
            detail = {
                **summary,
                "missing_in_artifact_examples": missing_examples,
                "extra_in_artifact_examples": extra_examples,
            }
            raise ValueError(
                "ColBERT window artifact is not aligned with DB cacheables:\n"
                + json.dumps(detail, ensure_ascii=False, indent=2)
            )
        return summary
    finally:
        connection.close()
        temporary_path.unlink(missing_ok=True)


def _window_bounded_region_index_specs(
    cacheable_ids: list[str],
    window_ids_by_cacheable_id: dict[str, list[str]],
) -> list[tuple[int, tuple[int, ...]]]:
    chunk_index_by_id = {
        str(cacheable_id): idx for idx, cacheable_id in enumerate(cacheable_ids)
    }
    raw_specs: list[tuple[int, tuple[int, ...]]] = []
    for center_idx, cacheable_id in enumerate(cacheable_ids):
        window_ids = window_ids_by_cacheable_id.get(str(cacheable_id), [cacheable_id])
        selected_indices = tuple(
            sorted(
                {
                    chunk_index_by_id[str(window_id)]
                    for window_id in window_ids
                    if str(window_id) in chunk_index_by_id
                }
            )
        )
        if not selected_indices:
            continue
        raw_specs.append((center_idx, selected_indices))

    selected_sets = [selected for _, selected in raw_specs]
    final_specs = []
    seen = set()
    for center_idx, selected_indices in raw_specs:
        if selected_indices in seen:
            continue
        selected_set = set(selected_indices)
        if any(
            len(selected_indices) < len(other) and selected_set.issubset(other)
            for other in selected_sets
        ):
            continue
        seen.add(selected_indices)
        final_specs.append((center_idx, selected_indices))
    return final_specs


def add_region_specs_to_colbert_window_data(
    data_dir: str | Path,
    db_dir: str | Path,
    region_token_budget: int,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Attach query-independent, retrieval-bounded region membership metadata."""

    data_path = Path(data_dir)
    index_path = data_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing ColBERT data index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") == colbert_artifact.DATA_ARTIFACT_FORMAT:
        if (
            index.get("region_token_budget") == int(region_token_budget)
            and int(index.get("region_spec_chunk_count", 0)) > 0
        ):
            return index
        raise ValueError(
            "finalized ColBERT split-JSON metadata cannot rebuild regions in "
            "place; rebuild the candidate store with overwrite enabled"
        )
    if index.get("format") != "colbert_window_data_build_v1":
        raise ValueError(
            f"unsupported ColBERT build data format: {index.get('format')}"
        )
    if (
        not overwrite
        and index.get("region_token_budget") == int(region_token_budget)
        and int(index.get("region_spec_chunk_count", 0)) > 0
    ):
        return index

    source_index_path = data_path.parent / "index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    window_token_budget = source_index.get("window_token_budget")
    if not isinstance(window_token_budget, int) or isinstance(
        window_token_budget, bool
    ):
        raise ValueError("ColBERT artifact requires an integer window_token_budget")
    if int(region_token_budget) != window_token_budget:
        raise ValueError(
            "region token budget must match the artifact window token budget: "
            f"region={region_token_budget}, window={window_token_budget}"
        )

    start_time = time.perf_counter()
    chunk_count = 0
    region_count = 0
    metadata_file = index.get("metadata_file")
    if not isinstance(metadata_file, str) or not metadata_file:
        raise ValueError("ColBERT data requires a metadata_file")
    metadata_path = data_path / metadata_file
    metadata_reader = ColBERTMetadataReader(metadata_path)
    metadata_writer = ColBERTMetadataWriter(metadata_path, create=False)
    metadata_writer.clear_regions()
    pending_groups = []

    def flush_groups() -> None:
        nonlocal chunk_count, region_count
        if not pending_groups:
            return
        unique_ids = list(
            dict.fromkeys(
                cacheable.id
                for _, cacheables in pending_groups
                for cacheable in cacheables
            )
        )
        windows = metadata_reader.window_ids_for_cacheable_ids(unique_ids)
        windows_by_id = dict(zip(unique_ids, windows))
        region_records = []
        for chunk_id, cacheables in pending_groups:
            cacheable_ids = [cacheable.id for cacheable in cacheables]
            specs = _window_bounded_region_index_specs(
                cacheable_ids=cacheable_ids,
                window_ids_by_cacheable_id=windows_by_id,
            )
            region_records.append((str(chunk_id), cacheable_ids, specs))
            chunk_count += 1
            region_count += len(specs)
        metadata_writer.replace_regions(region_records)
        pending_groups.clear()

    try:
        for chunk_id, chunk_cacheables in iter_db_cacheable_groups(db_dir):
            filtered = [cacheable for cacheable in chunk_cacheables if cacheable.text]
            if not filtered:
                continue
            pending_groups.append((str(chunk_id), filtered))
            if len(pending_groups) >= 2048:
                flush_groups()
        flush_groups()
    finally:
        metadata_reader.close()
        metadata_writer.close()

    index["region_token_budget"] = int(region_token_budget)
    index["region_spec_chunk_count"] = chunk_count
    index["region_spec_count"] = region_count
    index["region_spec_build_time_sec"] = time.perf_counter() - start_time
    index["region_spec_reuse_mode"] = "document_window_bound"
    export_reader = ColBERTMetadataReader(metadata_path)
    try:
        index.update(write_split_metadata_from_sqlite(export_reader, data_path))
    finally:
        export_reader.close()
    index["format"] = colbert_artifact.DATA_ARTIFACT_FORMAT
    index.pop("metadata_file", None)
    index.pop("region_spec_validation_mismatch_count", None)
    index.pop("region_spec_tokenizer", None)
    index.pop("region_spec_doc_token_overhead", None)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_path.unlink()
    return index
