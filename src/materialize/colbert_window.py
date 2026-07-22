from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from chunk import CacheableChunk
from materialize.db_manifest import (
    DB_BUILD_MANIFEST_FILENAME,
    build_db_manifest_reference,
    db_build_manifest_sha256,
    read_referenced_db_manifest,
)
from materialize.splitter.base import DocumentSplitter

ARTIFACT_FORMAT = "colbert_window_artifact_v1"
DATA_ARTIFACT_FORMAT = "colbert_window_data_v1"


def default_colbert_repo_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "third_party" / "ColBERT")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool value: {value!r}")


def import_official_colbert(repo_path: str, disable_cpu_extension: bool = True):
    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"official ColBERT repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        import ujson  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["ujson"] = json

    if disable_cpu_extension:
        from colbert.modeling.colbert import ColBERT

        ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)

    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    return Checkpoint, ColBERTConfig


@dataclass
class WindowSpec:
    text: str
    center_start: int
    center_end: int
    selected_indices: list[int]
    addition_order: list[int]
    truncated_center: bool


class _TokenizerOnlyModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _SentenceViewSplitter(DocumentSplitter):
    def build_chunks(self, filename: str, text: str, token_ids: list[int]):
        raise NotImplementedError


def _document_title(text: str) -> str:
    for line in text.splitlines():
        title = line.strip()
        if title:
            return title
    return ""


def _prefix_window_with_title(
    spec: WindowSpec, title: str, separator: str
) -> WindowSpec:
    if not title:
        return spec
    prefix = f"{title} {separator} "
    return WindowSpec(
        text=prefix + spec.text,
        center_start=spec.center_start + len(prefix),
        center_end=spec.center_end + len(prefix),
        selected_indices=spec.selected_indices,
        addition_order=spec.addition_order,
        truncated_center=spec.truncated_center,
    )


def _title_prefix_text(title: str, separator: str) -> str:
    return f"{title} {separator} " if title else ""


def _normalize_title_separator(separator: Any) -> str:
    if isinstance(separator, str):
        return separator
    if isinstance(separator, list) and len(separator) == 1:
        return f"[{separator[0]}]"
    return str(separator)


def _insert_marker_offsets(offsets: torch.Tensor) -> torch.Tensor:
    marker_offsets = torch.zeros(
        (offsets.size(0), 1, offsets.size(2)),
        dtype=offsets.dtype,
        device=offsets.device,
    )
    return torch.cat([offsets[:, :1], marker_offsets, offsets[:, 1:]], dim=1)


class ColBERTWindowEncoder:
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
        if repo_path is None:
            raise ValueError(
                "repo_path is required; do not use /tmp for the official ColBERT repo"
            )
        Checkpoint, ColBERTConfig = import_official_colbert(
            repo_path, disable_cpu_extension
        )
        config_kwargs: dict[str, Any] = {"checkpoint": model_name}
        if doc_maxlen is not None:
            config_kwargs["doc_maxlen"] = int(doc_maxlen)
        if query_maxlen is not None:
            config_kwargs["query_maxlen"] = int(query_maxlen)
        if attend_to_mask_tokens is not None:
            config_kwargs["attend_to_mask_tokens"] = bool(attend_to_mask_tokens)
        if mask_punctuation is not None:
            config_kwargs["mask_punctuation"] = bool(mask_punctuation)
        config = ColBERTConfig(**config_kwargs)
        if device == "cpu":
            config.gpus = 0
        self.checkpoint = Checkpoint(model_name, colbert_config=config, verbose=0)
        if not self.checkpoint.colbert_config.mask_punctuation and not hasattr(
            self.checkpoint, "skiplist"
        ):
            self.checkpoint.skiplist = {}
        self.model_name = model_name
        self.repo_path = repo_path
        self.device = device
        self.batch_size = batch_size
        self.verify_tensorization = verify_tensorization
        self.doc_maxlen = int(self.checkpoint.doc_tokenizer.doc_maxlen)
        self.query_maxlen = int(self.checkpoint.query_tokenizer.query_maxlen)
        self.dim = int(self.checkpoint.colbert_config.dim)
        self.max_length = int(max_length or self.doc_maxlen)
        if self.max_length > self.doc_maxlen:
            raise ValueError(
                "COLBERT_WINDOW_TOKEN_BUDGET cannot exceed the official ColBERT doc_maxlen: "
                f"budget={self.max_length}, doc_maxlen={self.doc_maxlen}"
            )
        self.doc_tokenizer = self.checkpoint.doc_tokenizer.tok
        self.doc_marker_id = int(self.checkpoint.doc_tokenizer.D_marker_token_id)
        self.doc_token_overhead = (
            len(
                self.doc_tokenizer(
                    [""],
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                    verbose=False,
                )["input_ids"][0]
            )
            + 1
        )

    def token_count(self, text: str) -> int:
        encoded = self.doc_tokenizer(
            [text],
            padding=False,
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )
        return len(encoded["input_ids"][0]) + 1

    def token_counts(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.doc_tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )
        return [len(input_ids) + 1 for input_ids in encoded["input_ids"]]

    def token_counts_without_specials(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.doc_tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            verbose=False,
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

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
        self, sentences: list[str], token_budget: int
    ) -> list[WindowSpec]:
        if not sentences:
            return []

        if token_budget > self.doc_maxlen:
            raise ValueError(
                "ColBERT window token budget cannot exceed official doc_maxlen: "
                f"budget={token_budget}, doc_maxlen={self.doc_maxlen}"
            )

        sentence_token_counts = self.token_counts_without_specials(sentences)
        specs: list[WindowSpec | None] = [None] * len(sentences)
        states: list[dict[str, Any] | None] = []

        for center_idx, center_text in enumerate(sentences):
            center_token_count = (
                sentence_token_counts[center_idx] + self.doc_token_overhead
            )
            if center_token_count >= token_budget:
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
                if token_count > token_budget:
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

    def tensorize_docs_with_offsets(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.doc_tokenizer(
            texts,
            padding="longest",
            truncation="longest_first",
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=self.doc_maxlen - 1,
        )
        raw_offsets = encoded.pop("offset_mapping")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        marker_ids = torch.full(
            (input_ids.size(0), 1),
            self.doc_marker_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        ids = torch.cat([input_ids[:, :1], marker_ids, input_ids[:, 1:]], dim=1)
        mask = torch.cat(
            [
                attention_mask[:, :1],
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask[:, 1:],
            ],
            dim=1,
        )
        offsets = _insert_marker_offsets(raw_offsets)
        return ids, mask, offsets

    def _verify_official_tensorization(
        self, texts: list[str], ids: torch.Tensor, mask: torch.Tensor
    ) -> None:
        if not self.verify_tensorization:
            return
        official_ids, official_mask = self.checkpoint.doc_tokenizer.tensorize(texts)
        official_ids = official_ids.detach().cpu()
        official_mask = official_mask.detach().cpu()
        if not torch.equal(ids.detach().cpu(), official_ids):
            mismatch_rows = (
                (ids.detach().cpu() != official_ids)
                .any(dim=1)
                .nonzero()
                .flatten()
                .tolist()
            )
            raise AssertionError(
                f"offset-aware doc tensorization ids differ from official rows={mismatch_rows[:10]}"
            )
        if not torch.equal(mask.detach().cpu(), official_mask):
            mismatch_rows = (
                (mask.detach().cpu() != official_mask)
                .any(dim=1)
                .nonzero()
                .flatten()
                .tolist()
            )
            raise AssertionError(
                f"offset-aware doc tensorization mask differs from official rows={mismatch_rows[:10]}"
            )

    @staticmethod
    def _center_positions(
        offsets: torch.Tensor,
        doc_mask: torch.Tensor,
        center_start: int,
        center_end: int,
    ) -> list[int]:
        selected: list[int] = []
        fallback: list[int] = []
        for token_idx, ((begin, end), valid) in enumerate(
            zip(offsets.tolist(), doc_mask.tolist())
        ):
            if not valid or (begin == 0 and end == 0):
                continue
            fallback.append(token_idx)
            if end > center_start and begin < center_end:
                selected.append(token_idx)
        return selected or fallback

    def encode_windows(
        self, specs: list[WindowSpec], show_progress: bool = False
    ) -> list[torch.Tensor]:
        vectors: list[torch.Tensor] = []
        iterator = range(0, len(specs), self.batch_size)
        for start in tqdm(
            iterator,
            desc="encode official colbert windows",
            disable=not show_progress,
            leave=False,
        ):
            batch_specs = specs[start : start + self.batch_size]
            texts = [spec.text for spec in batch_specs]
            ids, mask, offsets = self.tensorize_docs_with_offsets(texts)
            self._verify_official_tensorization(texts, ids, mask)
            with torch.no_grad():
                doc_vectors, doc_mask = self.checkpoint.doc(
                    ids,
                    mask,
                    keep_dims="return_mask",
                    to_cpu=True,
                )
            doc_mask = doc_mask.squeeze(-1).detach().cpu()
            offsets = offsets.detach().cpu()
            for row_idx, spec in enumerate(batch_specs):
                positions = self._center_positions(
                    offsets[row_idx],
                    doc_mask[row_idx],
                    int(spec.center_start),
                    int(spec.center_end),
                )
                if positions:
                    vectors.append(
                        doc_vectors[row_idx, positions].contiguous().to(torch.float16)
                    )
                else:
                    vectors.append(torch.empty((0, self.dim), dtype=torch.float16))
        return vectors

    def encode_queries(self, queries: list[str]) -> list[torch.Tensor]:
        query_vectors = self.checkpoint.queryFromText(
            queries,
            bsize=self.batch_size,
            to_cpu=True,
        )
        return [row.contiguous().to(torch.float32) for row in query_vectors]


def _iter_document_files(docs_dir: Path) -> Iterable[Path]:
    return (path for path in sorted(docs_dir.iterdir()) if path.is_file())


def _document_id_from_cacheable_id(cacheable_id: str) -> str | None:
    if "::" in cacheable_id:
        return cacheable_id.split("::", 1)[0]
    marker = ".txt-"
    marker_index = cacheable_id.find(marker)
    if marker_index >= 0:
        return cacheable_id[: marker_index + len(".txt")]
    return None


def db_document_ids(db_dir: str | Path, batch_size: int = 2048) -> set[str]:
    doc_ids: set[str] = set()
    for cacheable in _iter_db_cacheables(db_dir=str(db_dir), batch_size=batch_size):
        doc_id = _document_id_from_cacheable_id(str(cacheable.id))
        if doc_id is not None:
            doc_ids.add(doc_id)
    return doc_ids


def build_colbert_window_artifact(
    docs_dir: str,
    output_dir: str,
    db_dir: str,
    model_name: str = "colbert-ir/colbertv2.0",
    device: str | None = None,
    batch_size: int = 32,
    window_token_budget: int = 0,
    overwrite: bool = False,
    repo_path: str | None = None,
    disable_cpu_extension: bool = True,
    verify_tensorization: bool = True,
    mask_punctuation: bool | None = None,
    center_unit: str = "sentence",
    fixed_chunk_size: int | None = None,
    prefix_title: bool = False,
    title_separator: str = "[SEP]",
    include_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    title_separator = _normalize_title_separator(title_separator)
    if center_unit not in {
        "sentence",
        "sentence_only",
        "fixed_chunk",
        "fixed_chunk_window",
    }:
        raise ValueError(
            "center_unit must be one of "
            "{'sentence', 'sentence_only', 'fixed_chunk', 'fixed_chunk_window'}, "
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
    db_manifest, db_manifest_reference = build_db_manifest_reference(
        db_dir=db_dir, artifact_dir=artifact_dir
    )
    source_tokenizer_name = db_manifest.get("tokenizer_name")
    if not isinstance(source_tokenizer_name, str) or not source_tokenizer_name:
        raise ValueError("DB build manifest tokenizer_name must be a non-empty string")
    max_subchunk_tokens = db_manifest.get("max_subchunk_tokens")
    if max_subchunk_tokens is not None and (
        isinstance(max_subchunk_tokens, bool)
        or not isinstance(max_subchunk_tokens, int)
    ):
        raise ValueError(
            "DB build manifest max_subchunk_tokens must be an integer or null, "
            f"got {max_subchunk_tokens!r}"
        )
    repo_path = repo_path or default_colbert_repo_path()
    if overwrite and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifact_dir / "index.json"
    data_index_path = data_dir / "index.json"
    if index_path.exists() and data_index_path.exists() and not overwrite:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                "existing ColBERT artifact was built by an unsupported/non-official implementation; "
                "set COLBERT_WINDOW_OVERWRITE=True to rebuild it"
            )
        existing_reference = existing.get("db_manifest")
        if not isinstance(existing_reference, dict):
            raise ValueError("existing ColBERT v2 artifact is missing db_manifest")
        read_referenced_db_manifest(
            artifact_dir=artifact_dir, reference=existing_reference
        )
        if existing_reference.get("sha256") != db_manifest_reference["sha256"]:
            raise ValueError(
                "existing ColBERT artifact does not match the requested DB manifest"
            )
        data_existing = json.loads(data_index_path.read_text(encoding="utf-8"))
        if data_existing.get("format") != DATA_ARTIFACT_FORMAT:
            raise ValueError(
                "existing ColBERT data has unsupported format: "
                f"{data_existing.get('format')}"
            )
        return existing

    start_time = time.perf_counter()
    source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_name)
    visible_token_overhead = len(
        source_tokenizer.encode("", add_special_tokens=False)
    ) + len(source_tokenizer.encode("\n\n", add_special_tokens=False))
    fixed_body_chunk_size = None
    if center_unit in {"fixed_chunk", "fixed_chunk_window"}:
        fixed_body_chunk_size = int(fixed_chunk_size) - visible_token_overhead
        if fixed_body_chunk_size <= 0:
            raise ValueError(
                "fixed_chunk_size must be larger than prompt-visible token overhead: "
                f"fixed_chunk_size={fixed_chunk_size}, overhead={visible_token_overhead}"
            )
    splitter = _SentenceViewSplitter(
        docs_dir=str(docs_path),
        model=_TokenizerOnlyModel(source_tokenizer),
        cacheable_chunk_size=None,
        retrievable_chunk_size=None,
        content_chunk_size=None,
        max_subchunk_tokens=max_subchunk_tokens,
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

    def build_fixed_chunk_window_spec(
        chunk_start: int,
        chunk_end: int,
        center_text: str,
        token_budget: int,
    ) -> WindowSpec:
        if center_unit != "fixed_chunk_window":
            return WindowSpec(
                text=center_text,
                center_start=0,
                center_end=len(center_text),
                selected_indices=[len(units) - 1],
                addition_order=[len(units) - 1],
                truncated_center=False,
            )

        center_len = max(0, chunk_end - chunk_start)
        context_budget = max(0, token_budget - center_len - visible_token_overhead)
        left_budget = context_budget // 2
        right_budget = context_budget - left_budget
        left_start = max(0, chunk_start - left_budget)
        right_end = min(len(token_ids), chunk_end + right_budget)
        left_tokens = token_ids[left_start:chunk_start]
        right_tokens = token_ids[chunk_end:right_end]

        def make_text() -> tuple[str, int, int]:
            parts: list[str] = []
            cursor = 0
            center_start = 0
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
            return "".join(parts), center_start, center_end

        window_text, center_start, center_end = make_text()
        return WindowSpec(
            text=window_text,
            center_start=center_start,
            center_end=center_end,
            selected_indices=[len(units) - 1],
            addition_order=[len(units) - 1],
            truncated_center=False,
        )

    vectors_file = "vectors.fp16.bin"
    offsets_file = "offsets.npy"
    vectors_path = data_dir / vectors_file

    index_docs: dict[str, dict[str, Any]] = {}
    offsets: list[int] = []
    id_to_row: dict[str, int] = {}
    window_ids_by_row: list[list[str]] = []
    total_cacheables = 0
    total_center_tokens = 0
    truncated_centers = 0
    skipped_existing = 0
    failed_docs: list[dict[str, str]] = []
    pending_docs: list[dict[str, Any]] = []
    pending_window_count = 0
    embedding_dim: int | None = None
    vector_handle = vectors_path.open("wb")

    def flush_pending_docs():
        nonlocal pending_docs
        nonlocal pending_window_count
        nonlocal total_cacheables
        nonlocal total_center_tokens
        nonlocal truncated_centers
        nonlocal embedding_dim

        if not pending_docs:
            return

        flat_specs = [
            spec for pending_doc in pending_docs for spec in pending_doc["specs"]
        ]
        flat_vectors = encoder.encode_windows(flat_specs) if flat_specs else []
        cursor = 0

        for pending_doc in pending_docs:
            specs = pending_doc["specs"]
            vectors = flat_vectors[cursor : cursor + len(specs)]
            cursor += len(specs)
            doc_token_counts = [int(vector.shape[0]) for vector in vectors]
            if vectors and embedding_dim is None:
                embedding_dim = int(vectors[0].shape[1])

            index_docs[pending_doc["doc_id"]] = {
                "cacheable_count": len(pending_doc["cacheable_ids"]),
                "center_token_count": sum(doc_token_counts),
            }
            for cacheable_id, vector, spec in zip(
                pending_doc["cacheable_ids"], vectors, specs
            ):
                row = len(id_to_row)
                id_to_row[str(cacheable_id)] = row
                offsets.append(total_center_tokens)
                if int(vector.shape[1]) != (embedding_dim or encoder.dim):
                    raise ValueError(
                        f"embedding dim mismatch for {cacheable_id}: "
                        f"{vector.shape[1]} != {embedding_dim or encoder.dim}"
                    )
                vector = vector.contiguous().to(torch.float16).cpu()
                vector_handle.write(vector.numpy().tobytes(order="C"))
                total_center_tokens += int(vector.shape[0])
                cacheable_ids = pending_doc["cacheable_ids"]
                window_ids_by_row.append(
                    [
                        str(cacheable_ids[idx])
                        for idx in spec.selected_indices
                        if isinstance(idx, int) and 0 <= idx < len(cacheable_ids)
                    ]
                )
            total_cacheables += len(pending_doc["cacheable_ids"])
            truncated_centers += sum(1 for spec in specs if spec.truncated_center)

        pending_docs = []
        pending_window_count = 0

    doc_files = list(_iter_document_files(docs_path))
    if include_doc_ids is not None:
        doc_files = [path for path in doc_files if path.name in include_doc_ids]
    for path in tqdm(doc_files, desc="build colbert window artifact"):
        doc_id = path.name
        try:
            text = path.read_text(encoding="utf-8")
            title = _document_title(text) if prefix_title else ""
            title_prefix = _title_prefix_text(title, title_separator)
            title_prefix_tokens = (
                encoder.token_counts_without_specials([title_prefix])[0]
                if title_prefix
                else 0
            )
            content_token_budget = effective_token_budget - title_prefix_tokens
            if content_token_budget <= encoder.doc_token_overhead:
                raise ValueError(
                    "title prefix leaves no room for ColBERT window content: "
                    f"doc_id={doc_id}, budget={effective_token_budget}, "
                    f"title_prefix_tokens={title_prefix_tokens}"
                )
            token_ids = source_tokenizer.encode(text, add_special_tokens=False)
            if center_unit in {"sentence", "sentence_only"}:
                sentence_views = splitter._split_long_sentence_views(
                    splitter._build_sentence_views(text, token_ids), token_ids
                )
                units = [
                    view.text.strip() for view in sentence_views if view.text.strip()
                ]
                cacheable_ids = [f"{doc_id}::sent_{idx}" for idx in range(len(units))]
                if center_unit == "sentence_only":
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
                        token_budget=content_token_budget,
                    )
                if title:
                    specs = [
                        _prefix_window_with_title(spec, title, title_separator)
                        for spec in specs
                    ]
            else:
                if fixed_body_chunk_size is None:
                    raise RuntimeError("fixed_body_chunk_size was not initialized")
                units = []
                cacheable_ids = []
                specs = []
                for chunk_start in range(0, len(token_ids), fixed_body_chunk_size):
                    chunk_tokens = token_ids[
                        chunk_start : chunk_start + fixed_body_chunk_size
                    ]
                    if not chunk_tokens:
                        continue
                    chunk_text = source_tokenizer.decode(
                        chunk_tokens, skip_special_tokens=True
                    )
                    if not chunk_text:
                        continue
                    cacheable_ids.append(f"{doc_id}-{chunk_start}")
                    units.append(chunk_text)
                    specs.append(
                        build_fixed_chunk_window_spec(
                            chunk_start=chunk_start,
                            chunk_end=chunk_start + len(chunk_tokens),
                            center_text=chunk_text,
                            token_budget=content_token_budget,
                        )
                    )
                if title:
                    specs = [
                        _prefix_window_with_title(spec, title, title_separator)
                        for spec in specs
                    ]
        except Exception as exc:
            failed_docs.append({"doc_id": doc_id, "error": repr(exc)})
            continue

        pending_docs.append(
            {
                "doc_id": doc_id,
                "sentences": units,
                "cacheable_ids": cacheable_ids,
                "specs": specs,
            }
        )
        pending_window_count += len(specs)
        if pending_window_count >= batch_size:
            flush_pending_docs()

    flush_pending_docs()
    vector_handle.close()
    final_offsets = np.asarray(offsets + [total_center_tokens], dtype=np.int64)
    np.save(data_dir / offsets_file, final_offsets)
    embedding_dim = int(embedding_dim or encoder.dim)

    summary = {
        "format": ARTIFACT_FORMAT,
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
        "prefix_title": bool(prefix_title),
        "title_separator": title_separator,
        "center_unit": center_unit,
        "fixed_chunk_size": fixed_chunk_size,
        "fixed_body_chunk_size": fixed_body_chunk_size,
        "num_docs": len(index_docs),
        "num_input_docs": len(doc_files),
        "num_cacheables": total_cacheables,
        "num_center_tokens": total_center_tokens,
        "avg_center_tokens_per_cacheable": (
            total_center_tokens / total_cacheables if total_cacheables else 0.0
        ),
        "truncated_centers": truncated_centers,
        "skipped_existing": skipped_existing,
        "failed_docs": failed_docs,
        "build_time_sec": time.perf_counter() - start_time,
        "docs": index_docs,
    }
    index_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    data_summary = {
        "format": DATA_ARTIFACT_FORMAT,
        "source_format": summary["format"],
        "embedding_dim": embedding_dim,
        "num_cacheables": total_cacheables,
        "num_tokens": total_center_tokens,
        "vectors_file": vectors_file,
        "offsets_file": offsets_file,
        "id_to_row": id_to_row,
        "window_ids_by_row": window_ids_by_row,
        "build_time_sec": summary["build_time_sec"],
    }
    data_index_path.write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def global_top_count(candidate_count: int, rate: float) -> int:
    if candidate_count <= 0:
        return 0
    return min(candidate_count, max(1, math.ceil(candidate_count * rate)))


def score_maxsim(query_vectors: torch.Tensor, doc_vectors: torch.Tensor) -> float:
    if query_vectors.numel() == 0 or doc_vectors.numel() == 0:
        return float("-inf")
    sims = torch.matmul(
        query_vectors.to(torch.float32), doc_vectors.to(torch.float32).T
    )
    return float(sims.max(dim=1).values.sum().item())


class ColBERTWindowData:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"missing ColBERT data index: {self.index_path}")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != DATA_ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported ColBERT data format: " f"{self.index.get('format')}"
            )
        self.embedding_dim = int(self.index["embedding_dim"])
        self.num_tokens = int(self.index["num_tokens"])
        self.id_to_row = self.index["id_to_row"]
        self.window_ids_by_row = self.index.get("window_ids_by_row", [])
        self.region_token_budget = self.index.get("region_token_budget")
        if not isinstance(self.region_token_budget, int) or isinstance(
            self.region_token_budget, bool
        ):
            raise ValueError("ColBERT data requires an integer region_token_budget")
        self.region_specs_by_chunk = self.index.get("region_specs_by_chunk")
        if not isinstance(self.region_specs_by_chunk, dict):
            raise ValueError("ColBERT data requires region_specs_by_chunk")
        self.offsets = np.load(
            self.data_dir / self.index["offsets_file"], mmap_mode="r"
        )
        vectors_path = self.data_dir / self.index["vectors_file"]
        self.vectors = np.memmap(
            vectors_path,
            dtype=np.float16,
            mode="r",
            shape=(self.num_tokens, self.embedding_dim),
        )
        self.empty = torch.empty((0, self.embedding_dim), dtype=torch.float16)

    def vectors_for_cacheable_ids(self, cacheable_ids) -> list[torch.Tensor]:
        vectors = []
        for cacheable_id in cacheable_ids:
            row = self.id_to_row.get(cacheable_id)
            if row is None:
                vectors.append(self.empty)
                continue
            start = int(self.offsets[row])
            end = int(self.offsets[row + 1])
            if end <= start:
                vectors.append(self.empty)
                continue
            array = self.vectors[start:end]
            vectors.append(torch.from_numpy(array).to(torch.float16))
        return vectors

    def window_ids_for_cacheable_ids(self, cacheable_ids) -> list[list[str]]:
        window_ids = []
        for cacheable_id in cacheable_ids:
            row = self.id_to_row.get(cacheable_id)
            if row is None or row >= len(self.window_ids_by_row):
                window_ids.append([cacheable_id])
            else:
                ids = self.window_ids_by_row[row]
                window_ids.append(ids if ids else [cacheable_id])
        return window_ids

    def region_specs_for_doc(self, doc, token_budget: int):
        if self.region_token_budget != int(token_budget):
            raise ValueError(
                "runtime region budget does not match ColBERT artifact: "
                f"runtime={token_budget}, artifact={self.region_token_budget}"
            )
        doc_id = str(getattr(doc, "id", ""))
        payload = self.region_specs_by_chunk.get(doc_id)
        if payload is None:
            raise ValueError(f"ColBERT region specs missing runtime chunk: {doc_id}")
        cacheable_ids = [
            getattr(cacheable, "id", None)
            for cacheable in getattr(doc, "cacheables", []) or []
            if getattr(cacheable, "text", None)
        ]
        if payload.get("cacheable_ids") != cacheable_ids:
            raise ValueError(
                "runtime cacheable IDs do not match ColBERT region specs: "
                f"chunk={doc_id}"
            )
        return [
            (int(item[0]), tuple(int(idx) for idx in item[1]))
            for item in payload.get("specs", [])
        ]


class ColBERTWindowArtifact:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.index_path = self.artifact_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"missing ColBERT window artifact index: {self.index_path}"
            )
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        artifact_format = self.index.get("format")
        if artifact_format != ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported ColBERT window artifact format: "
                f"{artifact_format}; rebuild with the official ColBERT path"
            )
        self.db_manifest_reference = self.index.get("db_manifest")
        if not isinstance(self.db_manifest_reference, dict):
            raise ValueError("ColBERT artifact requires a DB manifest reference")
        self.db_manifest = read_referenced_db_manifest(
            artifact_dir=self.artifact_dir,
            reference=self.db_manifest_reference,
        )
        self.retrievable_vectors_cache: dict[
            tuple[str, tuple[str | None, ...]], list[torch.Tensor]
        ] = {}
        data_dir = self.index.get("data_dir")
        if not isinstance(data_dir, str) or not data_dir:
            raise ValueError("ColBERT artifact requires a data_dir")
        self.data = ColBERTWindowData(self.artifact_dir / data_dir)
        window_token_budget = self.index.get("window_token_budget")
        if not isinstance(window_token_budget, int) or isinstance(
            window_token_budget, bool
        ):
            raise ValueError("ColBERT artifact requires an integer window_token_budget")
        if self.data.region_token_budget != window_token_budget:
            raise ValueError(
                "ColBERT region budget does not match artifact window budget: "
                f"region={self.data.region_token_budget}, window={window_token_budget}"
            )

    def validate_db_manifest(self, db_dir: str | Path) -> None:
        referenced_path = (
            self.artifact_dir / self.db_manifest_reference["path"]
        ).resolve()
        runtime_path = (Path(db_dir) / DB_BUILD_MANIFEST_FILENAME).resolve()
        if runtime_path != referenced_path:
            raise ValueError(
                "runtime DB manifest path does not match ColBERT artifact reference: "
                f"artifact={referenced_path}, runtime={runtime_path}"
            )
        expected_sha256 = self.db_manifest_reference["sha256"]
        actual_sha256 = db_build_manifest_sha256(db_dir)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "runtime DB build manifest does not match ColBERT artifact: "
                f"artifact={expected_sha256}, runtime={actual_sha256}, db_dir={db_dir}"
            )

    def vectors_for_doc(self, doc) -> list[torch.Tensor]:
        cacheable_ids = tuple(
            getattr(cacheable, "id", None)
            for cacheable in getattr(doc, "cacheables", []) or []
        )
        cache_key = (str(getattr(doc, "id", "")), cacheable_ids)
        cached = self.retrievable_vectors_cache.get(cache_key)
        if cached is not None:
            return cached
        vectors = self.data.vectors_for_cacheable_ids(cacheable_ids)
        self.retrievable_vectors_cache[cache_key] = vectors
        return vectors

    def window_cacheable_ids_for_doc(self, doc) -> list[list[str]]:
        cacheable_ids = tuple(
            getattr(cacheable, "id", None)
            for cacheable in getattr(doc, "cacheables", []) or []
        )
        return self.data.window_ids_for_cacheable_ids(cacheable_ids)

    def region_specs_for_doc(self, doc, token_budget: int):
        return self.data.region_specs_for_doc(doc, token_budget)


def _iter_db_cacheables(
    db_dir: str, batch_size: int = 2048
) -> Iterable[CacheableChunk]:
    sqlite_path = Path(db_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"missing Chroma sqlite database: {sqlite_path}")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        total_records = connection.execute(
            "select count(*) from embedding_metadata where key='cacheables_json'"
        ).fetchone()[0]
        query = (
            "select cacheables.string_value "
            "from embeddings e "
            "join embedding_metadata cacheables "
            "  on e.id = cacheables.id and cacheables.key = 'cacheables_json' "
            "order by e.id "
            "limit ? offset ?"
        )
        for offset in range(0, total_records, batch_size):
            rows = connection.execute(query, (batch_size, offset)).fetchall()
            for (cacheables_json,) in rows:
                payload = json.loads(cacheables_json) if cacheables_json else []
                for item in payload:
                    if isinstance(item, dict):
                        yield CacheableChunk.from_payload(item)
    finally:
        connection.close()


def _iter_db_cacheable_groups(
    db_dir: str | Path, batch_size: int = 2048
) -> Iterable[tuple[str, list[CacheableChunk]]]:
    sqlite_path = Path(db_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"missing Chroma sqlite database: {sqlite_path}")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        total_records = connection.execute(
            "select count(*) from embedding_metadata where key='cacheables_json'"
        ).fetchone()[0]
        query = (
            "select e.embedding_id, cacheables.string_value "
            "from embeddings e "
            "join embedding_metadata cacheables "
            "  on e.id = cacheables.id and cacheables.key = 'cacheables_json' "
            "order by e.id "
            "limit ? offset ?"
        )
        for offset in range(0, total_records, batch_size):
            rows = connection.execute(query, (batch_size, offset)).fetchall()
            for chunk_id, cacheables_json in rows:
                payload = json.loads(cacheables_json) if cacheables_json else []
                cacheables = [
                    CacheableChunk.from_payload(item)
                    for item in payload
                    if isinstance(item, dict)
                ]
                yield str(chunk_id), cacheables
    finally:
        connection.close()


def validate_colbert_window_artifact_against_db(
    artifact_dir: str,
    db_dir: str,
    batch_size: int = 2048,
) -> dict[str, Any]:
    artifact = ColBERTWindowArtifact(artifact_dir)
    artifact_ids = set(str(cacheable_id) for cacheable_id in artifact.data.id_to_row)

    db_text_by_id: dict[str, str] = {}
    duplicate_db_ids = 0
    for cacheable in _iter_db_cacheables(db_dir=db_dir, batch_size=batch_size):
        if cacheable.id in db_text_by_id:
            duplicate_db_ids += 1
            if db_text_by_id[cacheable.id] != cacheable.text:
                raise ValueError(
                    f"DB has conflicting text for duplicate cacheable id: {cacheable.id}"
                )
            continue
        db_text_by_id[cacheable.id] = cacheable.text

    missing_in_artifact = sorted(set(db_text_by_id) - artifact_ids)
    extra_in_artifact = sorted(artifact_ids - set(db_text_by_id))
    text_mismatches = []

    summary = {
        "db_dir": str(db_dir),
        "artifact_dir": str(artifact_dir),
        "db_cacheable_count": len(db_text_by_id),
        "artifact_cacheable_count": len(artifact_ids),
        "duplicate_db_ids": duplicate_db_ids,
        "missing_in_artifact_count": len(missing_in_artifact),
        "extra_in_artifact_count": len(extra_in_artifact),
        "text_mismatch_count": len(text_mismatches),
        "text_validation": "not_stored",
    }

    if missing_in_artifact or extra_in_artifact or text_mismatches:
        detail = {
            **summary,
            "missing_in_artifact_examples": missing_in_artifact[:10],
            "extra_in_artifact_examples": extra_in_artifact[:10],
            "text_mismatch_examples": text_mismatches[:10],
        }
        raise ValueError(
            "ColBERT window artifact is not aligned with DB cacheables:\n"
            + json.dumps(detail, ensure_ascii=False, indent=2)
        )

    return summary


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
    data_path = Path(data_dir)
    index_path = data_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing ColBERT data index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != DATA_ARTIFACT_FORMAT:
        raise ValueError(f"unsupported ColBERT data format: {index.get('format')}")
    if (
        not overwrite
        and index.get("region_token_budget") == int(region_token_budget)
        and index.get("region_specs_by_chunk")
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
    region_specs_by_chunk = {}
    chunk_count = 0
    region_count = 0
    window_ids_by_cacheable_id = {}
    id_by_row = {
        int(row): str(cacheable_id)
        for cacheable_id, row in index.get("id_to_row", {}).items()
    }
    for row_idx, window_ids in enumerate(index.get("window_ids_by_row", [])):
        cacheable_id = id_by_row.get(row_idx)
        if cacheable_id is None:
            continue
        window_ids_by_cacheable_id[cacheable_id] = [str(item) for item in window_ids]
    for chunk_id, chunk_cacheables in _iter_db_cacheable_groups(db_dir):
        filtered = [cacheable for cacheable in chunk_cacheables if cacheable.text]
        if not filtered:
            continue
        cacheable_ids = [cacheable.id for cacheable in filtered]
        specs = _window_bounded_region_index_specs(
            cacheable_ids=cacheable_ids,
            window_ids_by_cacheable_id=window_ids_by_cacheable_id,
        )
        region_specs_by_chunk[str(chunk_id)] = {
            "cacheable_ids": cacheable_ids,
            "specs": [[center_idx, list(selected)] for center_idx, selected in specs],
        }
        chunk_count += 1
        region_count += len(specs)

    index["region_token_budget"] = int(region_token_budget)
    index["region_specs_by_chunk"] = region_specs_by_chunk
    index["region_spec_chunk_count"] = chunk_count
    index["region_spec_count"] = region_count
    index["region_spec_build_time_sec"] = time.perf_counter() - start_time
    index["region_spec_reuse_mode"] = "document_window_bound"
    index.pop("region_spec_validation_mismatch_count", None)
    index.pop("region_spec_tokenizer", None)
    index.pop("region_spec_doc_token_overhead", None)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index
