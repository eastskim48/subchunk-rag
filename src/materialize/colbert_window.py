from __future__ import annotations

import json
import math
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from chunk import CacheableChunk
from materialize.splitter.base import DocumentSplitter

ARTIFACT_FORMAT = "matkv_official_colbert_doc_window_v1"


def default_colbert_repo_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "third_party" / "ColBERT")


def safe_doc_filename(doc_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in doc_id) + ".pt"


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

    def encode_full_windows(
        self, specs: list[WindowSpec], show_progress: bool = False
    ) -> list[torch.Tensor]:
        vectors: list[torch.Tensor] = []
        iterator = range(0, len(specs), self.batch_size)
        for start in tqdm(
            iterator,
            desc="encode official colbert full windows",
            disable=not show_progress,
            leave=False,
        ):
            batch_specs = specs[start : start + self.batch_size]
            texts = [spec.text for spec in batch_specs]
            ids, mask, _ = self.tensorize_docs_with_offsets(texts)
            self._verify_official_tensorization(texts, ids, mask)
            with torch.no_grad():
                doc_vectors, doc_mask = self.checkpoint.doc(
                    ids,
                    mask,
                    keep_dims="return_mask",
                    to_cpu=True,
                )
            doc_mask = doc_mask.squeeze(-1).detach().cpu()
            for row_idx in range(len(batch_specs)):
                positions = doc_mask[row_idx].nonzero().flatten().tolist()
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


def build_colbert_window_artifact(
    docs_dir: str,
    output_dir: str,
    source_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
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
) -> dict[str, Any]:
    title_separator = _normalize_title_separator(title_separator)
    if center_unit not in {"sentence", "fixed_chunk", "fixed_chunk_window"}:
        raise ValueError(
            "center_unit must be one of {'sentence', 'fixed_chunk', 'fixed_chunk_window'}, "
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
    repo_path = repo_path or default_colbert_repo_path()
    docs_out_dir = artifact_dir / "docs"
    if overwrite and docs_out_dir.exists():
        shutil.rmtree(docs_out_dir)
    docs_out_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifact_dir / "index.json"
    if index_path.exists() and not overwrite:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                "existing ColBERT artifact was built by an unsupported/non-official implementation; "
                "set COLBERT_WINDOW_OVERWRITE=True to rebuild it"
            )

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

    index_docs: dict[str, dict[str, Any]] = {}
    total_cacheables = 0
    total_center_tokens = 0
    truncated_centers = 0
    skipped_existing = 0
    failed_docs: list[dict[str, str]] = []
    pending_docs: list[dict[str, Any]] = []
    pending_window_count = 0

    def flush_pending_docs():
        nonlocal pending_docs
        nonlocal pending_window_count
        nonlocal total_cacheables
        nonlocal total_center_tokens
        nonlocal truncated_centers

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
            token_counts = [int(vector.shape[0]) for vector in vectors]
            payload = {
                "format": f"{ARTIFACT_FORMAT}_doc",
                "doc_id": pending_doc["doc_id"],
                "cacheable_ids": pending_doc["cacheable_ids"],
                "cacheable_texts": pending_doc["sentences"],
                "embedding_dim": int(vectors[0].shape[1]) if vectors else encoder.dim,
                "window_texts": [spec.text for spec in specs],
                "window_selected_indices": [spec.selected_indices for spec in specs],
                "window_addition_order": [spec.addition_order for spec in specs],
                "window_truncated_center": [spec.truncated_center for spec in specs],
                "center_token_vectors": vectors,
            }
            torch.save(payload, pending_doc["out_file"])

            index_docs[pending_doc["doc_id"]] = {
                "file": f"docs/{pending_doc['out_file'].name}",
                "cacheable_count": len(pending_doc["cacheable_ids"]),
                "center_token_count": sum(token_counts),
            }
            total_cacheables += len(pending_doc["cacheable_ids"])
            total_center_tokens += sum(token_counts)
            truncated_centers += sum(1 for spec in specs if spec.truncated_center)

        pending_docs = []
        pending_window_count = 0

    doc_files = list(_iter_document_files(docs_path))
    for path in tqdm(
        doc_files, desc="build official colbert document-window artifacts"
    ):
        doc_id = path.name
        out_file = docs_out_dir / safe_doc_filename(doc_id)
        if out_file.exists() and not overwrite:
            payload = torch.load(out_file, map_location="cpu")
            if payload.get("format") != f"{ARTIFACT_FORMAT}_doc":
                raise ValueError(
                    f"existing doc artifact was built by an unsupported/non-official implementation: {out_file}"
                )
            token_counts = [
                int(vector.shape[0])
                for vector in payload.get("center_token_vectors", [])
            ]
            index_docs[doc_id] = {
                "file": f"docs/{out_file.name}",
                "cacheable_count": len(payload.get("cacheable_ids", [])),
                "center_token_count": sum(token_counts),
            }
            total_cacheables += len(payload.get("cacheable_ids", []))
            total_center_tokens += sum(token_counts)
            truncated_centers += sum(
                1 for item in payload.get("window_truncated_center", []) if item
            )
            skipped_existing += 1
            continue

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
            if center_unit == "sentence":
                sentence_views = splitter._build_sentence_views(text, token_ids)
                units = [
                    view.text.strip() for view in sentence_views if view.text.strip()
                ]
                cacheable_ids = [f"{doc_id}::sent_{idx}" for idx in range(len(units))]
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
                "out_file": out_file,
                "sentences": units,
                "cacheable_ids": cacheable_ids,
                "specs": specs,
            }
        )
        pending_window_count += len(specs)
        if pending_window_count >= batch_size:
            flush_pending_docs()

    flush_pending_docs()

    summary = {
        "format": ARTIFACT_FORMAT,
        "encoder_impl": "official_colbert_checkpoint",
        "docs_dir": str(docs_path),
        "source_tokenizer_name": source_tokenizer_name,
        "model_name": model_name,
        "checkpoint_name": model_name,
        "repo_path": repo_path,
        "device": encoder.device,
        "batch_size": batch_size,
        "window_token_budget": effective_token_budget,
        "official_doc_maxlen": encoder.doc_maxlen,
        "official_query_maxlen": encoder.query_maxlen,
        "mask_punctuation": bool(encoder.checkpoint.colbert_config.mask_punctuation),
        "embedding_dim": encoder.dim,
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


class ColBERTWindowArtifact:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.index_path = self.artifact_dir / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"missing ColBERT window artifact index: {self.index_path}"
            )
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported ColBERT window artifact format: "
                f"{self.index.get('format')}; rebuild with the official ColBERT path"
            )
        self.doc_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def parent_doc_id_for_retrievable(doc) -> str:
        parent_doc_id = getattr(doc, "metadata", {}).get("parent_doc_id")
        if isinstance(parent_doc_id, str) and parent_doc_id:
            return parent_doc_id
        for cacheable in getattr(doc, "cacheables", []) or []:
            if getattr(cacheable, "parent_doc_id", None):
                return cacheable.parent_doc_id
        if "::ret_" in doc.id:
            return doc.id.split("::ret_", 1)[0]
        return doc.id

    def load_doc(self, doc_id: str) -> dict[str, Any]:
        if doc_id not in self.doc_cache:
            docs = self.index.get("docs", {})
            if doc_id not in docs:
                raise KeyError(
                    f"ColBERT window artifact missing parent doc_id={doc_id}"
                )
            path = self.artifact_dir / docs[doc_id]["file"]
            self.doc_cache[doc_id] = torch.load(path, map_location="cpu")
        return self.doc_cache[doc_id]

    def vectors_for_doc(self, doc) -> list[torch.Tensor]:
        parent_doc_id = self.parent_doc_id_for_retrievable(doc)
        payload = self.load_doc(parent_doc_id)
        artifact_ids = payload.get("cacheable_ids", [])
        vector_by_id = dict(zip(artifact_ids, payload.get("center_token_vectors", [])))
        dim = int(payload.get("embedding_dim", self.index.get("embedding_dim", 0)))
        empty = torch.empty((0, dim), dtype=torch.float16)
        return [
            vector_by_id.get(cacheable.id, empty)
            for cacheable in getattr(doc, "cacheables", []) or []
        ]

    def window_cacheable_ids_for_doc(self, doc) -> list[list[str]]:
        parent_doc_id = self.parent_doc_id_for_retrievable(doc)
        payload = self.load_doc(parent_doc_id)
        artifact_ids = payload.get("cacheable_ids", [])
        window_selected_indices = payload.get("window_selected_indices", [])
        window_ids_by_id = {}
        for cacheable_id, selected_indices in zip(
            artifact_ids, window_selected_indices
        ):
            window_ids_by_id[cacheable_id] = [
                artifact_ids[idx]
                for idx in selected_indices
                if isinstance(idx, int) and 0 <= idx < len(artifact_ids)
            ]
        return [
            window_ids_by_id.get(cacheable.id, [cacheable.id])
            for cacheable in getattr(doc, "cacheables", []) or []
        ]


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


def validate_colbert_window_artifact_against_db(
    artifact_dir: str,
    db_dir: str,
    batch_size: int = 2048,
) -> dict[str, Any]:
    artifact = ColBERTWindowArtifact(artifact_dir)
    artifact_text_by_id: dict[str, str] = {}

    for doc_id in artifact.index.get("docs", {}):
        payload = artifact.load_doc(doc_id)
        for cacheable_id, text in zip(
            payload.get("cacheable_ids", []), payload.get("cacheable_texts", [])
        ):
            artifact_text_by_id[str(cacheable_id)] = str(text)

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

    missing_in_artifact = sorted(set(db_text_by_id) - set(artifact_text_by_id))
    extra_in_artifact = sorted(set(artifact_text_by_id) - set(db_text_by_id))
    text_mismatches = [
        {
            "cacheable_id": cacheable_id,
            "db_text": db_text_by_id[cacheable_id],
            "artifact_text": artifact_text_by_id[cacheable_id],
        }
        for cacheable_id in sorted(set(db_text_by_id) & set(artifact_text_by_id))
        if db_text_by_id[cacheable_id] != artifact_text_by_id[cacheable_id]
    ]

    summary = {
        "db_dir": str(db_dir),
        "artifact_dir": str(artifact_dir),
        "db_cacheable_count": len(db_text_by_id),
        "artifact_cacheable_count": len(artifact_text_by_id),
        "duplicate_db_ids": duplicate_db_ids,
        "missing_in_artifact_count": len(missing_in_artifact),
        "extra_in_artifact_count": len(extra_in_artifact),
        "text_mismatch_count": len(text_mismatches),
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
