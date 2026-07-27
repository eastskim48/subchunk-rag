"""Coreference-aware dynamic-programming unit grouping."""

import logging
import re
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm as _tqdm
from transformers import logging as transformers_logging

from materialize.splitter.grouper.dp import BaseDPGrouper

PRONOUN_SUBJECTS = {
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "him",
    "them",
}
PRONOUN_POSSESSIVES = {"his", "her", "their", "its"}
LEADING_TOKEN_RE = re.compile(r"^(\s*)([A-Za-z][A-Za-z'’-]*)(.*)$")


def _word_tokenize(sentence: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", sentence)


def _build_sentence_word_index(
    sentences: Sequence[str],
) -> tuple[list[str], list[tuple[int, int]], list[int]]:
    words = []
    sentence_word_spans = []
    word_to_sentence = []
    cursor = 0
    for sent_idx, sentence in enumerate(sentences):
        sent_words = _word_tokenize(sentence)
        start = cursor
        words.extend(sent_words)
        word_to_sentence.extend([sent_idx] * len(sent_words))
        cursor += len(sent_words)
        sentence_word_spans.append((start, cursor))
    return words, sentence_word_spans, word_to_sentence


def _normalize_clusters(prediction: Any) -> list[list[tuple[int, int]]]:
    if hasattr(prediction, "get_clusters"):
        clusters = prediction.get_clusters(as_strings=False)
    elif isinstance(prediction, dict) and "clusters" in prediction:
        clusters = prediction["clusters"]
    else:
        raise ValueError(
            "unsupported fastcoref prediction object: cannot extract clusters"
        )

    normalized = []
    for cluster in clusters:
        normalized_cluster = []
        for mention in cluster:
            if len(mention) != 2:
                continue
            normalized_cluster.append((int(mention[0]), int(mention[1])))
        if normalized_cluster:
            normalized.append(normalized_cluster)
    return normalized


def _run_fastcoref(words: Sequence[str], model) -> Any:
    predictions = model.predict(texts=[list(words)], is_split_into_words=True)
    if not predictions:
        raise ValueError("fastcoref returned no predictions")
    return predictions[0]


def _find_leading_pronoun_cluster(
    sent_start: int,
    sent_end: int,
    words: Sequence[str],
    mention_to_cluster: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    if sent_start >= sent_end:
        return None
    for mention, cluster in mention_to_cluster.items():
        mention_start, mention_end = mention
        if mention_start != sent_start:
            continue
        if mention_end < sent_start or mention_end >= sent_end:
            continue
        mention_tokens = words[mention_start : mention_end + 1]
        if not mention_tokens:
            continue
        mention_first = mention_tokens[0].lower()
        if mention_first in PRONOUN_SUBJECTS or mention_first in PRONOUN_POSSESSIVES:
            return cluster
    return None


def _silent_tqdm(*args, **kwargs):
    kwargs.setdefault("disable", True)
    return _tqdm(*args, **kwargs)


class CorefPronounDPGrouper(BaseDPGrouper):
    """Prefer spans that keep leading pronouns with antecedent context."""

    name = "coref_pronoun_dp_128"
    score_init = -(10**9)
    chunk_init = -(10**9)

    def __init__(
        self,
        tokenizer,
        token_budget: int | None = None,
        fastcoref_model_name: str | None = None,
    ):
        super().__init__(tokenizer=tokenizer, token_budget=token_budget)
        from fastcoref import FCoref
        import fastcoref.modeling as fastcoref_modeling
        import fastcoref.trainer as fastcoref_trainer
        from datasets.utils.logging import disable_progress_bar

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.fastcoref_model_name = fastcoref_model_name or "biu-nlp/f-coref"
        disable_progress_bar()
        transformers_logging.set_verbosity_error()
        logging.getLogger("fastcoref").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        fastcoref_modeling.tqdm = _silent_tqdm
        fastcoref_trainer.tqdm = _silent_tqdm
        self.coref_model = FCoref(
            device=device, model_name_or_path=self.fastcoref_model_name
        )

    def prepare_state(self, unit_texts: list[str]):
        words, sentence_word_spans, _ = _build_sentence_word_index(unit_texts)
        if not words:
            return {
                "is_pronoun_start": [False] * len(unit_texts),
                "antecedent_sentence": [None] * len(unit_texts),
            }

        prediction = _run_fastcoref(words, self.coref_model)
        clusters = _normalize_clusters(prediction)
        mention_to_cluster = {}
        for cluster in clusters:
            sorted_cluster = sorted(cluster)
            for mention in sorted_cluster:
                mention_to_cluster[mention] = sorted_cluster

        is_pronoun_start = []
        antecedent_sentence = []
        for sentence_index, sentence in enumerate(unit_texts):
            match = LEADING_TOKEN_RE.match(sentence)
            if not match:
                is_pronoun_start.append(False)
                antecedent_sentence.append(None)
                continue
            token = match.group(2).lower()
            if token not in PRONOUN_SUBJECTS and token not in PRONOUN_POSSESSIVES:
                is_pronoun_start.append(False)
                antecedent_sentence.append(None)
                continue

            sent_start, sent_end = sentence_word_spans[sentence_index]
            if sent_start >= sent_end:
                is_pronoun_start.append(True)
                antecedent_sentence.append(None)
                continue

            cluster = _find_leading_pronoun_cluster(
                sent_start, sent_end, words, mention_to_cluster
            )
            antecedent_idx = None
            if cluster:
                for mention_start, mention_end in cluster:
                    if mention_end >= sent_start:
                        break
                    mention_tokens = words[mention_start : mention_end + 1]
                    if not mention_tokens:
                        continue
                    mention_first = mention_tokens[0].lower()
                    if (
                        mention_first in PRONOUN_SUBJECTS
                        or mention_first in PRONOUN_POSSESSIVES
                    ):
                        continue
                    antecedent_idx = next(
                        (
                            prior_sent_idx
                            for prior_sent_idx, (prior_start, prior_end) in enumerate(
                                sentence_word_spans
                            )
                            if prior_start <= mention_start < prior_end
                        ),
                        None,
                    )
            is_pronoun_start.append(True)
            antecedent_sentence.append(antecedent_idx)

        return {
            "is_pronoun_start": is_pronoun_start,
            "antecedent_sentence": antecedent_sentence,
        }

    def score_span(self, start: int, end: int, state) -> float:
        span_gain = 0.0
        for sent_idx in range(start, end):
            if not state["is_pronoun_start"][sent_idx]:
                continue
            antecedent_idx = state["antecedent_sentence"][sent_idx]
            if antecedent_idx is not None and start <= antecedent_idx < sent_idx:
                span_gain += 1.0
            elif sent_idx == start:
                span_gain -= 1.0
        return span_gain

    def is_better_candidate(
        self, candidate_score, candidate_chunks, best_score, best_chunks
    ) -> bool:
        return candidate_score > best_score or (
            candidate_score == best_score and candidate_chunks > best_chunks
        )
