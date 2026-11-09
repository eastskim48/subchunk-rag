"""Load LateOn's published modules through the existing ColBERT interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

LATEON_MODEL = "lightonai/LateOn"
LATEON_REVISION = "62911e105059585d244384c7d17826e35f669c17"


class _TokenProjection(nn.Module):
    """Match PyLate Dense, including its separately learned residual projection."""

    def __init__(self, config: dict):
        super().__init__()
        if config["activation_function"] != "torch.nn.modules.linear.Identity":
            raise ValueError("unsupported LateOn projection activation")
        source, target = config["in_features"], config["out_features"]
        self.linear = nn.Linear(source, target, bias=config["bias"])
        self.use_residual = config["use_residual"]
        if self.use_residual and source != target:
            self.residual = nn.Linear(source, target, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = self.linear(values)
        if self.use_residual:
            result = result + (
                self.residual(values) if hasattr(self, "residual") else values
            )
        return result


class _LateOnTokenizer:
    def __init__(self, tokenizer, prefix: str, max_length: int):
        self.tok = tokenizer
        self.doc_maxlen = self.query_maxlen = max_length
        self.marker_id = tokenizer.convert_tokens_to_ids(prefix)
        if self.marker_id == tokenizer.unk_token_id:
            raise ValueError(f"LateOn prefix is missing from vocabulary: {prefix!r}")
        self.D_marker_token_id = self.marker_id

    def tokenize(self, texts: list[str], *, offsets: bool = False):
        # Sentence Transformers strips strings before tokenization. Restore source
        # character coordinates afterward so centered-span extraction stays valid.
        encoded = self.tok(
            [text.strip() for text in texts],
            padding="longest",
            truncation="longest_first",
            max_length=self.doc_maxlen - 1,
            return_tensors="pt",
            return_offsets_mapping=offsets,
        )
        for name in ("input_ids", "attention_mask"):
            values = encoded[name]
            prefix = torch.full(
                (len(texts), 1),
                self.marker_id if name == "input_ids" else 1,
                dtype=values.dtype,
            )
            encoded[name] = torch.cat([values[:, :1], prefix, values[:, 1:]], dim=1)
        if offsets:
            positions = encoded["offset_mapping"]
            for row, text in enumerate(texts):
                leading = len(text) - len(text.lstrip())
                valid = positions[row, :, 1] > positions[row, :, 0]
                positions[row, valid] += leading
            zeros = torch.zeros((len(texts), 1, 2), dtype=positions.dtype)
            encoded["offset_mapping"] = torch.cat(
                [positions[:, :1], zeros, positions[:, 1:]], dim=1
            )
        return encoded

    def tensorize(self, texts: list[str]):
        encoded = self.tokenize(texts)
        return encoded["input_ids"], encoded["attention_mask"]


class LateOnCheckpoint(nn.Module):
    """Provide normalized token vectors without installing or patching PyLate."""

    def __init__(
        self,
        device: str,
        doc_maxlen: int | None = None,
        query_maxlen: int | None = None,
        attend_to_mask_tokens: bool | None = None,
        mask_punctuation: bool | None = None,
    ):
        super().__init__()
        if attend_to_mask_tokens:
            raise ValueError("LateOn does not use query expansion tokens")
        snapshot = Path(
            snapshot_download(
                LATEON_MODEL,
                revision=LATEON_REVISION,
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            )
        )
        settings = json.loads(
            (snapshot / "config_sentence_transformers.json").read_text()
        )
        if settings["do_query_expansion"]:
            raise ValueError("unsupported LateOn query expansion configuration")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        # PyLate uses the mask token as padding even when query expansion is off.
        tokenizer.pad_token_id = tokenizer.mask_token_id
        self.bert = AutoModel.from_pretrained(
            snapshot, local_files_only=True, attn_implementation="sdpa"
        )
        self.projections = nn.ModuleList()
        modules = json.loads((snapshot / "modules.json").read_text())
        if len(modules) != 4 or any(
            module["type"] != "pylate.models.Dense.Dense" for module in modules[1:]
        ):
            raise ValueError("unsupported LateOn module layout")
        for module in modules[1:]:
            directory = snapshot / module["path"]
            config = json.loads((directory / "config.json").read_text())
            projection = _TokenProjection(config)
            load_model(projection, str(directory / "model.safetensors"), strict=True)
            self.projections.append(projection)
        self.colbert_config = SimpleNamespace(
            dim=config["out_features"],
            mask_punctuation=True if mask_punctuation is None else mask_punctuation,
        )
        self.doc_tokenizer = _LateOnTokenizer(
            tokenizer,
            settings["document_prefix"],
            settings["document_length"] if doc_maxlen is None else doc_maxlen,
        )
        self.query_tokenizer = _LateOnTokenizer(
            copy.deepcopy(tokenizer),
            settings["query_prefix"],
            settings["query_length"] if query_maxlen is None else query_maxlen,
        )
        for length in (
            self.doc_tokenizer.doc_maxlen,
            self.query_tokenizer.query_maxlen,
        ):
            if not 4 <= length <= self.bert.config.max_position_embeddings:
                raise ValueError("LateOn sequence length is outside backbone limits")
        self.skiplist = (
            [
                tokenizer.convert_tokens_to_ids(word)
                for word in settings["skiplist_words"]
            ]
            if self.colbert_config.mask_punctuation
            else []
        )
        self.model_revision = LATEON_REVISION
        self.to(device).eval()

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        values = self.bert(
            input_ids=ids.to(device), attention_mask=mask.to(device)
        ).last_hidden_state
        for projection in self.projections:
            values = projection(values)
        return torch.nn.functional.normalize(values, p=2, dim=-1)

    @torch.inference_mode()
    def doc(self, ids, mask, keep_dims="return_mask", to_cpu=False):
        if keep_dims != "return_mask":
            raise ValueError("LateOn adapter requires keep_dims='return_mask'")
        values = self._encode(ids, mask)
        valid = mask.to(device=values.device, dtype=torch.bool)
        for token_id in self.skiplist:
            valid &= ids.to(values.device) != token_id
        values = values * valid.unsqueeze(-1)
        if to_cpu:
            values, valid = values.cpu(), valid.cpu()
        return values, valid.unsqueeze(-1)

    @torch.inference_mode()
    def queryFromText(self, queries, bsize, to_cpu=False):
        result = []
        for start in range(0, len(queries), bsize):
            ids, mask = self.query_tokenizer.tensorize(queries[start : start + bsize])
            values = self._encode(ids, mask)
            for row, valid in zip(values, mask.bool()):
                vector = row[valid.to(row.device)].contiguous()
                result.append(vector.cpu() if to_cpu else vector)
        return result
