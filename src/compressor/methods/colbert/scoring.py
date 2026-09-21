"""Late-interaction scoring primitives used by ColBERT compressors."""

import numpy as np
import torch
from numba import njit, prange


@njit(cache=True)
def _make_2bit_query_residual_lut(query: np.ndarray, weights: np.ndarray) -> np.ndarray:
    query_length = query.shape[0]
    groups = query.shape[1] // 4
    table = np.empty((query_length, groups, 256), dtype=np.float32)
    for query_idx in range(query_length):
        for group_idx in range(groups):
            base = group_idx * 4
            for byte_value in range(256):
                value = np.float32(0.0)
                for offset in range(4):
                    shift = 7 - 2 * offset
                    bucket = ((byte_value >> shift) & 1) | (
                        ((byte_value >> (shift - 1)) & 1) << 1
                    )
                    value += query[query_idx, base + offset] * weights[bucket]
                table[query_idx, group_idx, byte_value] = value
    return table


@njit(cache=True, parallel=True)
def _direct_2bit_sentence_token_maxsim(
    query: np.ndarray,
    codes: np.ndarray,
    packed_residuals: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    token_norms: np.ndarray,
    sentence_offsets: np.ndarray,
) -> np.ndarray:
    table = _make_2bit_query_residual_lut(query, weights)
    sentence_count = sentence_offsets.shape[0] - 1
    query_length = query.shape[0]
    groups = query.shape[1] // 4
    scores = np.full((sentence_count, query_length), -np.inf, dtype=np.float32)
    for sentence_idx in prange(sentence_count):
        start = sentence_offsets[sentence_idx]
        end = sentence_offsets[sentence_idx + 1]
        for query_idx in range(query_length):
            best = np.float32(-np.inf)
            for token_idx in range(start, end):
                centroid_idx = codes[token_idx]
                value = np.float32(0.0)
                for group_idx in range(groups):
                    base = group_idx * 4
                    value += query[query_idx, base] * centroids[centroid_idx, base]
                    value += (
                        query[query_idx, base + 1] * centroids[centroid_idx, base + 1]
                    )
                    value += (
                        query[query_idx, base + 2] * centroids[centroid_idx, base + 2]
                    )
                    value += (
                        query[query_idx, base + 3] * centroids[centroid_idx, base + 3]
                    )
                    value += table[
                        query_idx, group_idx, packed_residuals[token_idx, group_idx]
                    ]
                value /= token_norms[token_idx]
                if value > best:
                    best = value
            scores[sentence_idx, query_idx] = best
    return scores


def direct_2bit_sentence_token_maxsim(
    query_vectors: torch.Tensor,
    codes: np.ndarray,
    packed_residuals: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    token_norms: np.ndarray,
    sentence_offsets: np.ndarray,
) -> torch.Tensor:
    """Score stored 2-bit vectors without reconstructing a float32 matrix."""

    query = np.ascontiguousarray(query_vectors.detach().cpu().numpy(), dtype=np.float32)
    scores = _direct_2bit_sentence_token_maxsim(
        query,
        np.ascontiguousarray(codes, dtype=np.int64),
        np.ascontiguousarray(packed_residuals, dtype=np.uint8),
        np.ascontiguousarray(centroids, dtype=np.float32),
        np.ascontiguousarray(weights, dtype=np.float32),
        np.ascontiguousarray(token_norms, dtype=np.float32),
        np.ascontiguousarray(sentence_offsets, dtype=np.int64),
    )
    return torch.from_numpy(scores)


def sentence_token_maxsim(
    query_vectors: torch.Tensor, sentence_vectors: list[torch.Tensor]
) -> torch.Tensor:
    """Return one document-token maximum per sentence and query token."""

    query_float = query_vectors.to(torch.float32)
    scores = torch.full(
        (len(sentence_vectors), query_float.shape[0]),
        float("-inf"),
        dtype=torch.float32,
        device=query_float.device,
    )
    nonempty_items = [
        (idx, vectors)
        for idx, vectors in enumerate(sentence_vectors)
        if vectors.numel() > 0
    ]
    if not nonempty_items or query_float.numel() == 0:
        return scores

    sentence_ids = torch.repeat_interleave(
        torch.tensor(
            [idx for idx, _ in nonempty_items],
            dtype=torch.long,
            device=query_float.device,
        ),
        torch.tensor(
            [int(vectors.shape[0]) for _, vectors in nonempty_items],
            dtype=torch.long,
            device=query_float.device,
        ),
    )
    all_vectors = torch.cat([vectors for _, vectors in nonempty_items], dim=0).to(
        query_float.device
    )
    similarities = torch.matmul(query_float, all_vectors.to(torch.float32).T)
    sentence_index = sentence_ids.unsqueeze(0).expand(query_float.shape[0], -1)
    scores.T.scatter_reduce_(
        1, sentence_index, similarities, reduce="amax", include_self=True
    )
    return scores


def aggregate_sentence_maxsim(
    sentence_scores: torch.Tensor, sentence_groups: list[list[int]]
) -> list[float]:
    """Aggregate sentence/query-token maxima into ColBERT MaxSim group scores."""

    if not sentence_groups:
        return []
    if sentence_scores.shape[1] == 0:
        return [float("-inf")] * len(sentence_groups)

    max_group_size = max((len(indices) for indices in sentence_groups), default=0)
    if max_group_size == 0:
        return [float("-inf")] * len(sentence_groups)

    padded_indices = [
        list(indices) + [-1] * (max_group_size - len(indices))
        for indices in sentence_groups
    ]
    group_index = torch.tensor(
        padded_indices, dtype=torch.long, device=sentence_scores.device
    )
    valid = group_index >= 0
    gathered = sentence_scores[group_index.clamp_min(0)]
    gathered = gathered.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    group_scores = gathered.max(dim=1).values.sum(dim=1)
    return [float(value) for value in group_scores.detach().cpu().tolist()]


def score_maxsim(query_vectors: torch.Tensor, doc_vectors: torch.Tensor) -> float:
    """Compute ColBERT MaxSim: max over document tokens, then sum over query tokens."""

    if query_vectors.numel() == 0 or doc_vectors.numel() == 0:
        return float("-inf")
    sims = torch.matmul(
        query_vectors.to(torch.float32), doc_vectors.to(torch.float32).T
    )
    return float(sims.max(dim=1).values.sum().item())
