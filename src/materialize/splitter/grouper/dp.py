"""Dynamic-programming base for token-budgeted contiguous grouping."""

from materialize.splitter.grouper.base import TokenBudgetGrouper
from materialize.splitter.types import ParsedUnit


class BaseDPGrouper(TokenBudgetGrouper):
    """Optimize a partition of ordered units under a per-group token budget."""

    score_init = -(10**18)
    chunk_init = 10**9

    def group(self, units: list[ParsedUnit]) -> list[list[int]]:
        unit_texts = [unit.text.strip() for unit in units]
        n = len(unit_texts)
        if n == 0:
            return []

        chunk_token_len = self._build_token_length_views(unit_texts)
        state = self.prepare_state(unit_texts)

        dp_score = [self.score_init] * (n + 1)
        dp_chunks = [self.chunk_init] * (n + 1)
        backptr = [-1] * (n + 1)
        dp_score[0] = 0.0
        dp_chunks[0] = 0

        for end in range(1, n + 1):
            best_score = self.score_init
            best_chunks = self.chunk_init
            best_start = -1
            for start in range(0, end):
                total_len = chunk_token_len(start, end)
                if total_len > self.token_budget and (end - start) >= 2:
                    continue
                if not self.is_valid_span(start, end, state, chunk_token_len):
                    continue
                span_gain = self.score_span(start, end, state)
                candidate_score = dp_score[start] + span_gain
                candidate_chunks = dp_chunks[start] + 1
                if self.is_better_candidate(
                    candidate_score, candidate_chunks, best_score, best_chunks
                ):
                    best_score = candidate_score
                    best_chunks = candidate_chunks
                    best_start = start

            if best_start < 0:
                raise ValueError(f"no valid DP partition for grouper={self.name}")
            dp_score[end] = best_score
            dp_chunks[end] = best_chunks
            backptr[end] = best_start

        spans = []
        cursor = n
        while cursor > 0:
            start = backptr[cursor]
            spans.append((start, cursor))
            cursor = start
        spans.reverse()
        return [list(range(start, end)) for start, end in spans]

    def prepare_state(self, unit_texts: list[str]):
        raise NotImplementedError

    def is_valid_span(self, start: int, end: int, state, chunk_token_len) -> bool:
        return True

    def score_span(self, start: int, end: int, state) -> float:
        raise NotImplementedError

    def is_better_candidate(
        self,
        candidate_score: float,
        candidate_chunks: int,
        best_score: float,
        best_chunks: int,
    ) -> bool:
        raise NotImplementedError
