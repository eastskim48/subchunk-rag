"""Sentence-max Gaussian selection using precomputed whole-chunk vectors."""

import json
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch

from chunk import CacheableChunk
from compressor.base import Compressor
from colbert_artifact import colbert_2bit_token_record_dtype
from encoder.colbert import ColBERTEncoder, default_colbert_repo_path
from materialize.colbert_residual import decompress_2bit_rows
from materialize.splitter.parser.sentence import SentenceParser


def gaussian_density(query, vectors, positions, token_count, sigma):
    weights, indices = (query @ vectors.T).max(dim=1)
    peaks = positions[indices].cpu().numpy()
    weights = weights.cpu().numpy().astype(np.float64)
    distance = np.arange(token_count)[:, None] - peaks[None, :]
    density = (
        weights[None, :]
        * np.exp(-distance.astype(np.float64) ** 2 / (2 * sigma * sigma))
    ).sum(axis=1)
    return torch.from_numpy(density)


def midpoint_sentence_scores(
    query, vectors, positions, sentence_indices, sigma, normalize=False, anchors=None
):
    """Aggregate sentence MaxSim first, then smooth in source-token coordinates."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    similarities = (query @ vectors.T).cpu().numpy()
    raw_scores, midpoints = [], []
    for indices in sentence_indices:
        matching = np.isin(positions, indices)
        if not matching.any():
            raise ValueError("Sentence has no native matching tokens")
        raw_scores.append(
            similarities[:, matching].max(axis=1).astype(np.float64).sum()
        )
        # Both endpoints are inclusive source-token indices, including punctuation.
        midpoints.append((float(indices[0]) + float(indices[-1])) / 2)
    midpoints = np.asarray(midpoints if anchors is None else anchors, dtype=np.float64)
    if midpoints.shape != (len(raw_scores),):
        raise ValueError("Anchor count must match sentence count")
    distances = midpoints[:, None] - midpoints[None, :]
    kernel = np.exp(-(distances**2) / (2 * sigma * sigma))
    scores = kernel @ np.asarray(raw_scores, dtype=np.float64)
    if normalize:
        scores = scores / kernel.sum(axis=1)
    return scores


def select_sentences(candidates, budget):
    selected, used = [], 0
    for item in sorted(candidates, key=lambda c: -c["score"]):
        if used + item["tokens"] > budget:
            break
        selected.append(item)
        used += item["tokens"]
    return sorted(selected, key=lambda c: (c["rank"], c["index"]))


def sentence_spans(text, parser):
    from blingfire import text_to_sentences_and_offsets

    stripped = text.strip()
    origin = len(text) - len(text.lstrip())
    sentences, spans = text_to_sentences_and_offsets(stripped)
    units = sentences.split("\n")
    assert len(units) == len(spans)
    merged = []
    for unit, (left, right) in zip(units, spans):
        if not unit.strip():
            continue
        left, right = origin + left, origin + right
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if merged and parser._should_merge_with_next(merged[-1][0], unit):
            previous, start, _ = merged.pop()
            merged.append((f"{previous} {unit.strip()}", start, right))
        else:
            merged.append((unit.strip(), left, right))
    assert [unit for unit, _, _ in merged] == parser._split_sentence_texts(text)
    return [(left, right) for _, left, right in merged]


class ColBERTGaussianCompressor(Compressor):
    def __init__(self):
        base = Path(os.environ["COLBERT_CHUNK_DIR"])
        stats = json.loads((base / "prepared.json").read_text())
        artifact = base / "chunk_2bit"
        complete = json.loads((artifact / "complete.json").read_text())
        self.scoring = os.environ.get("COLBERT_GAUSSIAN_SCORING", "token_density_max")
        if self.scoring not in {
            "token_density_max",
            "sentence_midpoint",
            "sentence_midpoint_normalized",
        }:
            raise ValueError(f"Unknown Gaussian scoring: {self.scoring}")
        self.sigma = (
            float(stats["mean_sentence_tokens"])
            if self.scoring in {"sentence_midpoint", "sentence_midpoint_normalized"}
            else int(math.floor(stats["mean_sentence_tokens"] / 2 + 0.5))
        )
        sigma_scale = float(os.environ.get("COLBERT_GAUSSIAN_SIGMA_SCALE", "1.0"))
        if not math.isfinite(sigma_scale) or sigma_scale <= 0:
            raise ValueError("Gaussian sigma scale must be finite and positive")
        self.sigma *= sigma_scale
        self.ratio = float(os.environ["RETAIN_TOKEN_RATIO"])
        if not 0 < self.ratio <= 1 or self.sigma <= 0:
            raise ValueError("Invalid ratio or corpus sigma")
        self.mapping = sqlite3.connect(
            f'file:{artifact / "mapping.sqlite3"}?mode=ro', uri=True
        )
        self.source = sqlite3.connect(
            f'file:{base / "chunks.sqlite3"}?mode=ro', uri=True
        )
        self.data = np.memmap(
            artifact / "token_records.bin",
            mode="r",
            dtype=colbert_2bit_token_record_dtype(complete["dim"]),
        )
        with np.load(artifact / "codec.npz") as codec:
            self.centroids = np.asarray(codec["centroids"], dtype=np.float32)
            self.weights = np.asarray(codec["weights"], dtype=np.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.encoder = ColBERTEncoder(
            model_name=complete["model"],
            device="cpu",
            batch_size=20,
            doc_maxlen=complete["doc_maxlen"],
            query_maxlen=complete["query_maxlen"],
            query_truncation_side="right",
            repo_path=default_colbert_repo_path(),
            verify_tensorization=True,
        )
        self.parser = SentenceParser(self.encoder.doc_tokenizer)
        print(
            f"Gaussian {self.scoring}: sigma={self.sigma}, ratio={self.ratio}, maxsim=cpu, density=numpy_cpu",
            flush=True,
        )

    def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
        queries = self.encoder.encode_queries(batch_queries)
        result = []
        with torch.inference_mode():
            for docs, query in zip(batch_top_k_docs, queries):
                query = query.to("cpu")
                candidates, total = [], 0
                for rank, doc in enumerate(docs):
                    rid, text = str(doc.id), doc.text
                    assert (
                        self.source.execute(
                            "select text from chunks where id=?", (rid,)
                        ).fetchone()[0]
                        == text
                    )
                    start, count, offsets, positions = self.mapping.execute(
                        "select start,count,offsets,positions from chunks where id=?",
                        (rid,),
                    ).fetchone()
                    if count == 0:
                        continue
                    offsets = np.frombuffer(offsets, dtype=np.int32).reshape(-1, 2)
                    positions = np.frombuffer(positions, dtype=np.int32).copy()
                    rows = self.data[start : start + count]
                    vectors = decompress_2bit_rows(
                        rows["code"],
                        rows["residuals"],
                        self.centroids,
                        self.weights,
                    ).to("cpu")
                    spans = sentence_spans(text, self.parser)
                    sentence_indices = [
                        np.flatnonzero((offsets[:, 0] < right) & (offsets[:, 1] > left))
                        for left, right in spans
                    ]
                    assert all(len(indices) for indices in sentence_indices), rid
                    if self.scoring in {
                        "sentence_midpoint",
                        "sentence_midpoint_normalized",
                    }:
                        valid = [
                            i
                            for i, indices in enumerate(sentence_indices)
                            if np.isin(positions, indices).any()
                        ]
                        values = midpoint_sentence_scores(
                            query,
                            vectors,
                            positions,
                            [sentence_indices[i] for i in valid],
                            self.sigma,
                            normalize=self.scoring == "sentence_midpoint_normalized",
                        )
                        scores = dict(zip(valid, values))
                    else:
                        density = (
                            gaussian_density(
                                query,
                                vectors,
                                torch.as_tensor(
                                    positions, device="cpu", dtype=torch.long
                                ),
                                len(offsets),
                                self.sigma,
                            )
                            .cpu()
                            .numpy()
                        )
                        scores = {
                            i: density[indices].max()
                            for i, indices in enumerate(sentence_indices)
                        }
                    assert np.isfinite(list(scores.values())).all()
                    total += len(offsets)
                    assignment = np.zeros(len(offsets), dtype=np.int32)
                    for index, ((left, right), indices) in enumerate(
                        zip(spans, sentence_indices)
                    ):
                        assignment[indices] += 1
                        if index not in scores:
                            continue
                        candidates.append(
                            dict(
                                rank=rank,
                                index=index,
                                tokens=len(indices),
                                score=float(scores[index]),
                                text=text[left:right],
                            )
                        )
                    # Boundary-crossing tokens count toward each intersecting sentence.
                    assert np.all(assignment >= 1), rid
                selected = select_sentences(candidates, math.ceil(total * self.ratio))
                output = []
                for rank, doc in enumerate(docs):
                    items = [item for item in selected if item["rank"] == rank]
                    if not items:
                        continue
                    cloned = doc.clone()
                    cloned.cacheables = [
                        CacheableChunk(
                            id=f'{doc.id}::gaussian_{item["index"]}',
                            text=item["text"],
                            parent_doc_id=str(doc.id),
                        )
                        for item in items
                    ]
                    output.append(cloned)
                result.append(output)
        return result
