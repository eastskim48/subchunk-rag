"""Official-ColBERT adapter for query and contextualized document encoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def default_colbert_repo_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "third_party" / "ColBERT")


def import_official_colbert(
    repo_path: str,
    device: str,
    disable_cpu_extension: bool = True,
):
    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"official ColBERT repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        import ujson  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["ujson"] = json

    import colbert.parameters

    official_device = torch.device(device)
    colbert.parameters.DEVICE = official_device
    for module_name in (
        "colbert.modeling.base_colbert",
        "colbert.modeling.colbert",
        "colbert.modeling.tokenization.doc_tokenization",
        "colbert.modeling.tokenization.query_tokenization",
    ):
        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None and hasattr(loaded_module, "DEVICE"):
            loaded_module.DEVICE = official_device

    if disable_cpu_extension:
        from colbert.modeling.colbert import ColBERT

        ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)

    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    return Checkpoint, ColBERTConfig


def _insert_marker_offsets(offsets: torch.Tensor) -> torch.Tensor:
    marker_offsets = torch.zeros(
        (offsets.size(0), 1, offsets.size(2)),
        dtype=offsets.dtype,
        device=offsets.device,
    )
    return torch.cat([offsets[:, :1], marker_offsets, offsets[:, 1:]], dim=1)


class ColBERTEncoder:
    """Share ColBERT tensorization and encoding across offline and online paths."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        doc_maxlen: int | None = None,
        query_maxlen: int | None = None,
        query_minlen: int | None = None,
        query_truncation_side: str = "right",
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
            repo_path,
            device=device,
            disable_cpu_extension=disable_cpu_extension,
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
        actual_device = next(self.checkpoint.parameters()).device
        requested_device = torch.device(device)
        if actual_device.type != requested_device.type:
            raise RuntimeError(
                "ColBERT checkpoint loaded on the wrong device: "
                f"requested={requested_device}, actual={actual_device}"
            )
        self._configure_checkpoint_inference(
            checkpoint=self.checkpoint,
            requested_device=requested_device,
        )
        if query_truncation_side not in {"left", "right"}:
            raise ValueError(
                "query_truncation_side must be 'left' or 'right', got "
                f"{query_truncation_side!r}"
            )
        self.checkpoint.query_tokenizer.tok.truncation_side = query_truncation_side
        if not self.checkpoint.colbert_config.mask_punctuation and not hasattr(
            self.checkpoint, "skiplist"
        ):
            self.checkpoint.skiplist = {}
        self.model_name = model_name
        self.repo_path = repo_path
        self.device = device
        self.actual_device = str(actual_device)
        self.batch_size = batch_size
        self.verify_tensorization = verify_tensorization
        self.doc_maxlen = int(self.checkpoint.doc_tokenizer.doc_maxlen)
        self.query_maxlen = int(self.checkpoint.query_tokenizer.query_maxlen)
        self.query_minlen = int(query_minlen) if query_minlen is not None else None
        if self.query_minlen is not None and not (
            4 <= self.query_minlen <= self.query_maxlen
        ):
            raise ValueError(
                "query_minlen must satisfy 4 <= query_minlen <= query_maxlen, got "
                f"query_minlen={self.query_minlen}, query_maxlen={self.query_maxlen}"
            )
        self.query_truncation_side = query_truncation_side
        max_position_embeddings = int(
            self.checkpoint.bert.config.max_position_embeddings
        )
        if self.query_maxlen > max_position_embeddings:
            raise ValueError(
                "query_maxlen exceeds the ColBERT backbone position limit: "
                f"query_maxlen={self.query_maxlen}, "
                f"max_position_embeddings={max_position_embeddings}"
            )
        self.dim = int(self.checkpoint.colbert_config.dim)
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

    @staticmethod
    def _configure_checkpoint_inference(checkpoint, requested_device) -> None:
        """Disable the official CUDA AMP context for an explicitly CPU checkpoint."""

        amp_manager = getattr(checkpoint, "amp_manager", None)
        if requested_device.type == "cpu" and amp_manager is not None:
            amp_manager.activated = False

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

    def encode_document_spans(
        self,
        texts: list[str],
        center_spans: list[tuple[int, int]],
        show_progress: bool = False,
    ) -> list[torch.Tensor]:
        if len(texts) != len(center_spans):
            raise ValueError(
                "texts and center_spans must have the same length: "
                f"texts={len(texts)}, center_spans={len(center_spans)}"
            )

        vectors: list[torch.Tensor] = []
        iterator = range(0, len(texts), self.batch_size)
        for start in tqdm(
            iterator,
            desc="encode official colbert documents",
            disable=not show_progress,
            leave=False,
        ):
            batch_texts = texts[start : start + self.batch_size]
            batch_spans = center_spans[start : start + self.batch_size]
            ids, mask, offsets = self.tensorize_docs_with_offsets(batch_texts)
            self._verify_official_tensorization(batch_texts, ids, mask)
            with torch.no_grad():
                doc_vectors, doc_mask = self.checkpoint.doc(
                    ids,
                    mask,
                    keep_dims="return_mask",
                    to_cpu=True,
                )
            doc_mask = doc_mask.squeeze(-1).detach().cpu()
            offsets = offsets.detach().cpu()
            for row_idx, (center_start, center_end) in enumerate(batch_spans):
                positions = self._center_positions(
                    offsets[row_idx],
                    doc_mask[row_idx],
                    int(center_start),
                    int(center_end),
                )
                if positions:
                    vectors.append(
                        doc_vectors[row_idx, positions].contiguous().to(torch.float16)
                    )
                else:
                    vectors.append(torch.empty((0, self.dim), dtype=torch.float16))
        return vectors

    def encode_queries(self, queries: list[str]) -> list[torch.Tensor]:
        with torch.inference_mode():
            query_vectors = self.checkpoint.queryFromText(
                queries,
                bsize=self.batch_size,
                to_cpu=True,
            )
        if self.query_minlen is None:
            query_vectors = query_vectors.to(torch.float32).contiguous()
            return list(query_vectors.unbind(0))

        encoded = self.checkpoint.query_tokenizer.tok(
            queries,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        content_lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        effective_lengths = [
            max(
                self.query_minlen,
                min(self.query_maxlen, content_length + 3),
            )
            for content_length in content_lengths
        ]
        return [
            row[:effective_length].contiguous().to(torch.float32)
            for row, effective_length in zip(query_vectors, effective_lengths)
        ]
