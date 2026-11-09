"""All-query-token relevance smoothed in expanded-chunk token coordinates."""

import hashlib
import json
import math
import os

import numpy as np
import torch

from compressor.methods.colbert.gaussian import (
    ColBERTGaussianCompressor,
    select_sentences,
    gaussian_density,
    midpoint_sentence_scores,
)
from materialize.colbert_residual import decompress_2bit_rows


def all_token_density(query, vectors, positions, token_count, sigma, normalize):
    """Exact full-support Gaussian convolution; no peak selection or FFT."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    # Both encoder output and reconstructed document vectors are L2-normalized.
    relevance = vectors.float().numpy() @ query.float().numpy().sum(axis=0)
    signal = np.zeros(token_count, dtype=np.float64)
    np.add.at(signal, positions, relevance)
    distance = np.arange(1 - token_count, token_count, dtype=np.float64)
    kernel = np.exp(-(distance**2) / (2 * sigma**2))
    density = np.convolve(signal, kernel, mode="full")[
        token_count - 1 : 2 * token_count - 1
    ]
    if normalize:
        mass = np.zeros(token_count, dtype=np.float64)
        np.add.at(mass, positions, 1.0)
        denominator = np.convolve(mass, kernel, mode="full")[
            token_count - 1 : 2 * token_count - 1
        ]
        density = np.divide(
            density, denominator, out=np.zeros_like(density), where=denominator > 0
        )
    return density


class ColBERTAllTokenGaussianCompressor(ColBERTGaussianCompressor):
    def __init__(self):
        # Reuse the codec/encoder initialization without changing older methods.
        super().__init__()
        self.sigma = float(os.environ["COLBERT_GAUSSIAN_SIGMA"])
        mode = os.environ["COLBERT_ALL_TOKEN_NORMALIZE"]
        if mode not in {"true", "false"}:
            raise ValueError("COLBERT_ALL_TOKEN_NORMALIZE must be true or false")
        self.normalize = mode == "true"
        self.expanded_scoring = os.environ.get(
            "COLBERT_EXPANDED_SCORING", "all_token_sum"
        )
        if self.expanded_scoring not in {
            "all_token_sum",
            "sentence_midpoint_normalized",
            "sentence_index_normalized",
            "peak_token_max",
        }:
            raise ValueError(f"Unknown expanded scoring: {self.expanded_scoring}")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Invalid sigma")
        print(
            f"Expanded Gaussian: scoring={self.expanded_scoring}, sigma={self.sigma}, "
            f"all_token_normalized={self.normalize}, full_support, expanded_split180, cpu",
            flush=True,
        )

    def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
        queries = self.encoder.encode_queries(batch_queries)
        results = []
        with torch.inference_mode():
            for docs, query in zip(batch_top_k_docs, queries, strict=True):
                candidates, total = [], 0
                for rank, doc in enumerate(docs):
                    rid = str(doc.id)
                    text = self.source.execute(
                        "select text from chunks where id=?", (rid,)
                    ).fetchone()[0]
                    geometry, original_hash = self.source.execute(
                        "select units,original_sha256 from geometry where id=?", (rid,)
                    ).fetchone()
                    assert (
                        hashlib.sha256(doc.text.encode()).hexdigest() == original_hash
                    ), rid
                    spans = json.loads(geometry)
                    assert len(spans) == len(doc.cacheables), rid
                    for span, unit in zip(spans, doc.cacheables, strict=True):
                        assert span["id"] == unit.id, rid
                        assert text[span["start"] : span["end"]] == unit.text, rid
                    start, count, offsets, positions = self.mapping.execute(
                        "select start,count,offsets,positions from chunks where id=?",
                        (rid,),
                    ).fetchone()
                    if count == 0:
                        continue
                    offsets = np.frombuffer(offsets, dtype=np.int32).reshape(-1, 2)
                    positions = np.frombuffer(positions, dtype=np.int32)
                    rows = self.data[start : start + count]
                    vectors = decompress_2bit_rows(
                        rows["code"], rows["residuals"], self.centroids, self.weights
                    ).cpu()
                    sentence_indices = [
                        np.flatnonzero(
                            (offsets[:, 0] < span["end"])
                            & (offsets[:, 1] > span["start"])
                        )
                        for span in spans
                    ]
                    valid = [
                        i
                        for i, indices in enumerate(sentence_indices)
                        if np.isin(positions, indices).any()
                    ]
                    if self.expanded_scoring in {
                        "sentence_midpoint_normalized",
                        "sentence_index_normalized",
                    }:
                        values = midpoint_sentence_scores(
                            query.cpu(),
                            vectors,
                            positions,
                            [sentence_indices[i] for i in valid],
                            self.sigma,
                            normalize=True,
                            anchors=(
                                valid
                                if self.expanded_scoring == "sentence_index_normalized"
                                else None
                            ),
                        )
                        scores = dict(zip(valid, values))
                    elif self.expanded_scoring == "peak_token_max":
                        density = gaussian_density(
                            query.cpu(),
                            vectors,
                            torch.from_numpy(positions.copy()),
                            len(offsets),
                            self.sigma,
                        ).numpy()
                        scores = {i: density[sentence_indices[i]].max() for i in valid}
                    else:
                        density = all_token_density(
                            query.cpu(),
                            vectors,
                            positions,
                            len(offsets),
                            self.sigma,
                            self.normalize,
                        )
                        scores = {i: density[sentence_indices[i]].sum() for i in valid}
                    assert np.isfinite(list(scores.values())).all(), rid
                    total += len(offsets)
                    for index in valid:
                        indices = sentence_indices[index]
                        candidates.append(
                            dict(
                                rank=rank,
                                index=index,
                                tokens=len(indices),
                                score=float(scores[index]),
                            )
                        )
                selected = select_sentences(candidates, math.ceil(total * self.ratio))
                output = []
                for rank, doc in enumerate(docs):
                    chosen = [c["index"] for c in selected if c["rank"] == rank]
                    if chosen:
                        cloned = doc.clone()
                        cloned.cacheables = [doc.cacheables[i].clone() for i in chosen]
                        output.append(cloned)
                results.append(output)
        return results
