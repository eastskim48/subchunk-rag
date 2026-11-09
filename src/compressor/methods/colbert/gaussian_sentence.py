"""Normalized midpoint smoothing over persisted sentence-encoded units."""

import math
import os

import numpy as np
import torch

from colbert_artifact import ColBERTWindowArtifact
from compressor.base import Compressor
from compressor.methods.colbert.gaussian import select_sentences
from encoder.colbert import ColBERTEncoder, default_colbert_repo_path


def unit_geometry(units, tokenizer):
    """Use all linked units, including complete text outside retrieval boundaries."""
    text = "\n\n".join(unit.text for unit in units)
    offsets = np.asarray(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )["offset_mapping"],
        dtype=np.int64,
    ).reshape(-1, 2)
    cursor, indices = 0, []
    for unit in units:
        end = cursor + len(unit.text)
        idx = np.flatnonzero((offsets[:, 0] < end) & (offsets[:, 1] > cursor))
        if not len(idx):
            raise ValueError(f"Persisted unit {unit.id} has no source tokens")
        indices.append(idx)
        cursor = end + 2
    return len(offsets), indices


def smooth_unit_scores(query, vectors, indices, sigma):
    valid = [i for i, v in enumerate(vectors) if len(v)]
    if not valid:
        return {}, valid
    joined = torch.cat([vectors[i].float() for i in valid])
    sim = (query.float() @ joined.T).numpy()
    raw, mids, cursor = [], [], 0
    for i in valid:
        end = cursor + len(vectors[i])
        raw.append(sim[:, cursor:end].max(axis=1).astype(np.float64).sum())
        mids.append((float(indices[i][0]) + float(indices[i][-1])) / 2)
        cursor = end
    mids = np.asarray(mids)
    kernel = np.exp(-((mids[:, None] - mids[None, :]) ** 2) / (2 * sigma**2))
    scores = (kernel @ np.asarray(raw)) / kernel.sum(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite sentence Gaussian score")
    return dict(zip(valid, scores)), valid


class ColBERTSentenceGaussianCompressor(Compressor):
    def __init__(self):
        self.sigma = float(os.environ["COLBERT_GAUSSIAN_SIGMA"])
        self.ratio = float(os.environ["RETAIN_TOKEN_RATIO"])
        if not math.isfinite(self.sigma) or self.sigma <= 0 or not 0 < self.ratio <= 1:
            raise ValueError("Invalid sigma or ratio")
        self.artifact = ColBERTWindowArtifact(os.environ["COLBERT_WINDOW_DIR"])
        self.artifact.validate_db_manifest(os.environ["DB_DIR"])
        index = self.artifact.index
        if index["center_unit"] != "subchunk_only":
            raise ValueError("Sentence encoding artifact required")
        # Load codec conversion once, rather than copying the full table per chunk.
        self.artifact.data.centroids = np.asarray(
            self.artifact.data.centroids, dtype=np.float32
        )
        self.artifact.data.bucket_weights = np.asarray(
            self.artifact.data.bucket_weights, dtype=np.float32
        )
        self.encoder = ColBERTEncoder(
            model_name=index["model_name"],
            device="cpu",
            batch_size=20,
            doc_maxlen=index["official_doc_maxlen"],
            query_maxlen=index["official_query_maxlen"],
            query_truncation_side="right",
            repo_path=default_colbert_repo_path(),
            disable_cpu_extension=True,
            verify_tensorization=False,
        )
        self.encoder.checkpoint.float()
        print(
            f'Sentence Gaussian: model={index["model_name"]}, sigma={self.sigma}, '
            f"ratio={self.ratio}, complete_persisted_units, expanded_source_token_budget, normalized=True, cpu_fp32",
            flush=True,
        )

    def clear_inter_batch_cache(self):
        self.artifact.retrievable_vectors_cache.clear()

    def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
        queries = self.encoder.encode_queries(batch_queries)
        result = []
        with torch.inference_mode():
            for docs, query in zip(batch_top_k_docs, queries):
                candidates, total = [], 0
                for rank, doc in enumerate(docs):
                    count, indices = unit_geometry(
                        doc.cacheables, self.encoder.doc_tokenizer
                    )
                    vectors = self.artifact.vectors_for_doc(doc)
                    scores, valid = smooth_unit_scores(
                        query.cpu(), vectors, indices, self.sigma
                    )
                    if not valid:
                        continue
                    total += count
                    for i in valid:
                        candidates.append(
                            dict(
                                rank=rank,
                                index=i,
                                tokens=len(indices[i]),
                                score=float(scores[i]),
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
                result.append(output)
        return result
