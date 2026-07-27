"""Late-interaction scoring primitives used by ColBERT compressors."""

import torch


def score_maxsim(query_vectors: torch.Tensor, doc_vectors: torch.Tensor) -> float:
    """Compute ColBERT MaxSim: max over document tokens, then sum over query tokens."""

    if query_vectors.numel() == 0 or doc_vectors.numel() == 0:
        return float("-inf")
    sims = torch.matmul(
        query_vectors.to(torch.float32), doc_vectors.to(torch.float32).T
    )
    return float(sims.max(dim=1).values.sum().item())
