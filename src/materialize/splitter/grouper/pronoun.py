"""Lightweight pronoun-aware dynamic-programming grouping."""

from materialize.splitter.grouper.dp import BaseDPGrouper

PRONOUN_LIKE_STARTS = {
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "its",
    "him",
    "them",
}


def sentence_start_token(sentence: str) -> str:
    pieces = sentence.strip().split()
    if not pieces:
        return ""
    return pieces[0].strip().lower().strip("\"'([{")


class PronounDPGrouper(BaseDPGrouper):
    """Penalize boundaries that strand a leading pronoun without context."""

    name = "pronoun_dp_128"
    score_init = -(10**9)
    chunk_init = -(10**9)

    def prepare_state(self, unit_texts: list[str]):
        is_pronoun_start = [
            sentence_start_token(text) in PRONOUN_LIKE_STARTS for text in unit_texts
        ]
        pronoun_prefix = [0]
        for value in is_pronoun_start:
            pronoun_prefix.append(pronoun_prefix[-1] + int(value))
        return {
            "is_pronoun_start": is_pronoun_start,
            "pronoun_prefix": pronoun_prefix,
        }

    def score_span(self, start: int, end: int, state) -> float:
        pronouns_after_start = (
            state["pronoun_prefix"][end] - state["pronoun_prefix"][start + 1]
        )
        return pronouns_after_start - (1 if state["is_pronoun_start"][start] else 0)

    def is_better_candidate(
        self, candidate_score, candidate_chunks, best_score, best_chunks
    ) -> bool:
        return candidate_score > best_score or (
            candidate_score == best_score and candidate_chunks > best_chunks
        )
