"""Late-interaction scoring primitives used by ColBERT compressors."""

import torch


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
