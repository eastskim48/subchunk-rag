"""Cost-constrained chunk combination selection with CARROT-style MCTS."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoModelForSequenceClassification

from chunk import CacheableChunk, RetrievableChunk
from compressor.base import Compressor
from compressor.token_budget import TokenBudgetMixin


@dataclass
class _MCTSNode:
    """One ordered chunk combination explored by Monte Carlo tree search."""

    state: tuple[int, ...]
    token_cost: int
    parent: _MCTSNode | None = None
    children: list[_MCTSNode] = field(default_factory=list)
    tried_indices: set[int] = field(default_factory=set)
    visits: int = 0
    total_reward: float = 0.0
    reranker_score: float | None = None

    def ucb(self, parent_visits: int, exploration: float, penalty: float, budget: int):
        if self.visits == 0:
            return float("inf")
        average_reward = self.total_reward / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visits
        )
        cost_term = penalty * (self.token_cost / budget) if budget > 0 else 0.0
        return average_reward + exploration_term - cost_term


class CARROTCompressor(TokenBudgetMixin, Compressor):
    """Select and order retrieved chunks, optionally under a hard token budget."""

    def __init__(self):
        super().__init__()
        self._initialize_token_budget(require_budget=False)
        self.model_name = os.getenv(
            "CARROT_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual"
        )
        self.device = os.getenv(
            "CARROT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_iterations = int(os.getenv("CARROT_MAX_ITERATIONS", "10"))
        self.exploration = float(os.getenv("CARROT_C", "2.4"))
        self.cost_penalty = float(os.getenv("CARROT_LAMBDA", "0.1"))
        self.soft_budget = int(os.getenv("CARROT_SOFT_BUDGET", "8192"))
        self.rerank_batch_size = int(os.getenv("CARROT_BATCH_SIZE", "32"))
        self.rerank_max_length = int(os.getenv("CARROT_MAX_LENGTH", "1024"))
        if self.max_iterations <= 0:
            raise ValueError("CARROT_MAX_ITERATIONS must be positive")
        if self.rerank_batch_size <= 0:
            raise ValueError("CARROT_BATCH_SIZE must be positive")
        if self.rerank_max_length <= 0:
            raise ValueError("CARROT_MAX_LENGTH must be positive")
        if self.exploration < 0:
            raise ValueError("CARROT_C must be non-negative")
        if self.cost_penalty < 0:
            raise ValueError("CARROT_LAMBDA must be non-negative")
        if self.soft_budget <= 0:
            raise ValueError("CARROT_SOFT_BUDGET must be positive")

        print(
            "CARROT compression enabled. "
            f"Initializing reranker: {self.model_name} on {self.device}"
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            use_flash_attn=False,
        ).to(self.device)
        self.model.eval()
        if not hasattr(self.model, "compute_score"):
            raise TypeError(
                "CARROT_MODEL_NAME must provide the Jina-compatible compute_score API"
            )

    @staticmethod
    def _document_text(doc: RetrievableChunk) -> str:
        text = getattr(doc, "text", None)
        return text.strip() if isinstance(text, str) and text.strip() else ""

    @staticmethod
    def _context_text(docs: List[RetrievableChunk], state: tuple[int, ...]) -> str:
        return "".join(
            f"{document_text}\n\n"
            for doc_idx in state
            for document_text in [CARROTCompressor._document_text(docs[doc_idx])]
            if document_text
        )

    @staticmethod
    def _build_output_document(doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        document_text = CARROTCompressor._document_text(doc)
        cloned.cacheables = (
            [CacheableChunk(id=f"{doc.id}::carrot", text=document_text)]
            if document_text
            else []
        )
        return cloned

    def _context_token_count(
        self, docs: List[RetrievableChunk], state: tuple[int, ...]
    ) -> int:
        context = self._context_text(docs, state)
        if not context:
            return 0
        return len(
            self.budget_tokenizer(
                context,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )

    def _retrieved_context_token_count(self, docs: List[RetrievableChunk]) -> int:
        state = tuple(idx for idx, doc in enumerate(docs) if self._document_text(doc))
        return self._context_token_count(docs, state)

    def _reranker_text(
        self, docs: List[RetrievableChunk], state: tuple[int, ...]
    ) -> str:
        return " ".join(
            text for text in (self._document_text(docs[idx]) for idx in state) if text
        )

    @staticmethod
    def _normalize_scores(raw_scores, expected: int) -> list[float]:
        if isinstance(raw_scores, torch.Tensor):
            values = raw_scores.detach().float().cpu().reshape(-1).tolist()
        elif isinstance(raw_scores, (int, float)):
            values = [float(raw_scores)]
        elif hasattr(raw_scores, "tolist"):
            values = raw_scores.tolist()
            if isinstance(values, (int, float)):
                values = [float(values)]
        else:
            values = list(raw_scores)
        values = [float(value) for value in values]
        if len(values) != expected:
            raise ValueError(
                "CARROT reranker returned an unexpected number of scores: "
                f"expected {expected}, got {len(values)}"
            )
        return values

    def _score_nodes(
        self,
        query: str,
        docs: List[RetrievableChunk],
        nodes: list[_MCTSNode],
    ) -> list[float]:
        scores = []
        for start in range(0, len(nodes), self.rerank_batch_size):
            batch = nodes[start : start + self.rerank_batch_size]
            pairs = [[query, self._reranker_text(docs, node.state)] for node in batch]
            with torch.no_grad():
                raw_scores = self.model.compute_score(
                    pairs,
                    max_length=self.rerank_max_length,
                )
            scores.extend(self._normalize_scores(raw_scores, len(batch)))
        return scores

    @staticmethod
    def _backpropagate(node: _MCTSNode, reward: float) -> None:
        current = node
        first = True
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            if first:
                current.reranker_score = reward
                first = False
            current = current.parent

    @staticmethod
    def _available_indices(
        node: _MCTSNode, candidate_indices: tuple[int, ...]
    ) -> list[int]:
        selected = set(node.state)
        return [idx for idx in candidate_indices if idx not in selected]

    def _fully_expanded(
        self, node: _MCTSNode, candidate_indices: tuple[int, ...]
    ) -> bool:
        available = self._available_indices(node, candidate_indices)
        return all(idx in node.tried_indices for idx in available)

    def _expand(
        self,
        node: _MCTSNode,
        docs: List[RetrievableChunk],
        candidate_indices: tuple[int, ...],
        hard_budget: int | None,
    ) -> list[_MCTSNode]:
        expanded = []
        for idx in self._available_indices(node, candidate_indices):
            if idx in node.tried_indices:
                continue
            node.tried_indices.add(idx)
            state = node.state + (idx,)
            token_cost = self._context_token_count(docs, state)
            if hard_budget is not None and token_cost > hard_budget:
                continue
            child = _MCTSNode(state=state, token_cost=token_cost, parent=node)
            node.children.append(child)
            expanded.append(child)
        return expanded

    @staticmethod
    def _collect_scored_nodes(root: _MCTSNode) -> list[_MCTSNode]:
        scored = []
        stack = list(root.children)
        while stack:
            node = stack.pop()
            if node.reranker_score is not None:
                scored.append(node)
            stack.extend(node.children)
        return scored

    def _search(
        self, docs: List[RetrievableChunk], query: str, hard_budget: int | None
    ) -> tuple[int, ...]:
        candidate_indices = tuple(
            idx for idx, doc in enumerate(docs) if self._document_text(doc)
        )
        if not candidate_indices or (hard_budget is not None and hard_budget <= 0):
            return ()

        root = _MCTSNode(state=(), token_cost=0)
        for _ in range(self.max_iterations):
            node = root
            while self._fully_expanded(node, candidate_indices) and node.children:
                node = max(
                    node.children,
                    key=lambda child: child.ucb(
                        parent_visits=node.visits,
                        exploration=self.exploration,
                        penalty=self.cost_penalty,
                        budget=self.soft_budget,
                    ),
                )

            expanded = self._expand(
                node=node,
                docs=docs,
                candidate_indices=candidate_indices,
                hard_budget=hard_budget,
            )
            if not expanded:
                continue
            rewards = self._score_nodes(query=query, docs=docs, nodes=expanded)
            for child, reward in zip(expanded, rewards):
                self._backpropagate(child, reward)

        scored_nodes = self._collect_scored_nodes(root)
        if hard_budget is not None:
            scored_nodes = [
                node for node in scored_nodes if node.token_cost <= hard_budget
            ]
        if not scored_nodes:
            return ()
        best = max(
            scored_nodes,
            key=lambda node: (
                node.reranker_score,
                node.visits,
                len(node.state),
            ),
        )
        return best.state

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        if len(batch_top_k_docs) != len(batch_queries):
            raise ValueError(
                "CARROT requires one retrieved-document batch per query: "
                f"got {len(batch_top_k_docs)} document batches and "
                f"{len(batch_queries)} queries"
            )

        selected_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            hard_budget = self._resolve_final_token_budget(docs)
            state = self._search(docs=docs, query=query, hard_budget=hard_budget)
            selected_batches.append(
                [self._build_output_document(docs[idx]) for idx in state]
            )
        return selected_batches
