from __future__ import annotations

import os
import re
import logging
from typing import Any, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm as _tqdm
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging


def _env_token_budget(default: int = 128) -> int:
    return int(os.getenv("TOKEN_BUDGET", str(default)))


def _env_similarity_threshold(default: float = 0.7) -> float:
    return float(os.getenv("SIMILARITY_THRESHOLD", str(default)))


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


def sentence_start_token(sentence: str) -> str:
    pieces = sentence.strip().split()
    if not pieces:
        return ""
    return pieces[0].strip().lower().strip("\"'([{")


def _word_tokenize(sentence: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", sentence)


def _build_sentence_word_index(
    sentences: Sequence[str],
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    words: List[str] = []
    sentence_word_spans: List[Tuple[int, int]] = []
    word_to_sentence: List[int] = []
    cursor = 0
    for sent_idx, sentence in enumerate(sentences):
        sent_words = _word_tokenize(sentence)
        start = cursor
        words.extend(sent_words)
        word_to_sentence.extend([sent_idx] * len(sent_words))
        cursor += len(sent_words)
        sentence_word_spans.append((start, cursor))
    return words, sentence_word_spans, word_to_sentence


def _normalize_clusters(prediction: Any) -> List[List[Tuple[int, int]]]:
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
    mention_to_cluster: dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[Tuple[int, int]] | None:
    if sent_start >= sent_end:
        return None
    # fastcoref mentions often cover the leading pronoun plus the next content
    # word, e.g. "His many" or "He won", so exact (sent_start, sent_start)
    # matching is too strict. Use the first mention that starts at the sentence
    # boundary and whose first token is a pronoun-like token.
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


class SubchunkMerger:
    name = "base"

    def __init__(self, tokenizer, token_budget: int | None = None):
        self.tokenizer = tokenizer
        self.token_budget = (
            _env_token_budget() if token_budget is None else token_budget
        )

    def merge(self, sentence_texts: List[str]) -> List[List[int]]:
        raise NotImplementedError

    def _build_token_length_views(self, sentence_texts: List[str]):
        first_token_lengths = [
            len(self.tokenizer.encode(text, add_special_tokens=False))
            for text in sentence_texts
        ]
        continued_token_lengths = [
            len(self.tokenizer.encode(f" {text}", add_special_tokens=False))
            for text in sentence_texts
        ]
        continued_prefix = [0]
        for value in continued_token_lengths:
            continued_prefix.append(continued_prefix[-1] + value)

        def chunk_token_len(start: int, end: int) -> int:
            if start >= end:
                return 0
            return first_token_lengths[start] + (
                continued_prefix[end] - continued_prefix[start + 1]
            )

        return chunk_token_len


class DPMerger(SubchunkMerger):
    score_init = -(10**18)
    chunk_init = 10**9

    def merge(self, sentence_texts: List[str]) -> List[List[int]]:
        n = len(sentence_texts)
        if n == 0:
            return []

        chunk_token_len = self._build_token_length_views(sentence_texts)
        state = self.prepare_state(sentence_texts)

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
                raise ValueError(f"no valid DP partition for merger={self.name}")
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

    def prepare_state(self, sentence_texts: List[str]):
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


class PronounDPMerger(DPMerger):
    name = "pronoun_dp_128"
    score_init = -(10**9)
    chunk_init = -(10**9)

    def prepare_state(self, sentence_texts: List[str]):
        is_pronoun_start = [
            sentence_start_token(text) in PRONOUN_LIKE_STARTS for text in sentence_texts
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


class CorefPronounDPMerger(DPMerger):
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

    def prepare_state(self, sentence_texts: List[str]):
        words, sentence_word_spans, _ = _build_sentence_word_index(sentence_texts)
        if not words:
            return {
                "is_pronoun_start": [False] * len(sentence_texts),
                "antecedent_sentence": [None] * len(sentence_texts),
            }

        prediction = _run_fastcoref(words, self.coref_model)
        clusters = _normalize_clusters(prediction)
        mention_to_cluster: dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for cluster in clusters:
            sorted_cluster = sorted(cluster)
            for mention in sorted_cluster:
                mention_to_cluster[mention] = sorted_cluster

        is_pronoun_start: List[bool] = []
        antecedent_sentence: List[int | None] = []
        for sentence_index, sentence in enumerate(sentence_texts):
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


class EmbeddingSimilarityMerger(DPMerger):
    name = "embed_sim_128"

    def __init__(
        self,
        tokenizer,
        token_budget: int | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 128,
        device: str | None = None,
    ):
        super().__init__(tokenizer=tokenizer, token_budget=token_budget)
        self.similarity_threshold = (
            _env_similarity_threshold()
            if similarity_threshold is None
            else similarity_threshold
        )
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.embed_model = AutoModel.from_pretrained(
            self.embedding_model,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.embed_model.eval()

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        encoded = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.embed_model(**encoded)

        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _embed_texts_batched(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            hidden_size = getattr(self.embed_model.config, "hidden_size", 0)
            return torch.empty((0, hidden_size), dtype=torch.float32)

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch_embeddings = self._embed_texts(texts[start : start + self.batch_size])
            batches.append(batch_embeddings.detach().cpu())
        return (
            torch.cat(batches, dim=0)
            if batches
            else torch.empty((0, 0), dtype=torch.float32)
        )

    def prepare_state(self, sentence_texts: List[str]):
        n = len(sentence_texts)
        sentence_embeddings = self._embed_texts_batched(sentence_texts).to(
            torch.float32
        )
        adjacent_similarities: list[float] = []
        for idx in range(n - 1):
            similarity = torch.dot(
                sentence_embeddings[idx], sentence_embeddings[idx + 1]
            ).item()
            adjacent_similarities.append(float(similarity))

        valid_span = [[False] * (n + 1) for _ in range(n)]
        span_similarity_gain = [[0.0] * (n + 1) for _ in range(n)]
        chunk_token_len = self._build_token_length_views(sentence_texts)
        for start in range(n):
            valid_span[start][start + 1] = True
            running_gain = 0.0
            for end in range(start + 2, n + 1):
                total_len = chunk_token_len(start, end)
                if total_len > self.token_budget:
                    break
                edge_similarity = adjacent_similarities[end - 2]
                if edge_similarity < self.similarity_threshold:
                    break
                running_gain += edge_similarity - self.similarity_threshold
                valid_span[start][end] = True
                span_similarity_gain[start][end] = running_gain

        return {
            "valid_span": valid_span,
            "span_similarity_gain": span_similarity_gain,
        }

    def is_valid_span(self, start: int, end: int, state, chunk_token_len) -> bool:
        if (end - start) < 2:
            return True
        return state["valid_span"][start][end]

    def score_span(self, start: int, end: int, state) -> float:
        if (end - start) < 2:
            return 0.0
        return state["span_similarity_gain"][start][end]

    def is_better_candidate(
        self, candidate_score, candidate_chunks, best_score, best_chunks
    ) -> bool:
        return candidate_score > best_score or (
            candidate_score == best_score and candidate_chunks < best_chunks
        )


def build_merger(name: str | None, tokenizer):
    if name is None:
        return None
    if name == "pronoun_dp":
        return PronounDPMerger(tokenizer=tokenizer)
    if name == "coref_pronoun_dp":
        return CorefPronounDPMerger(tokenizer=tokenizer)
    if name == "embed_sim":
        return EmbeddingSimilarityMerger(tokenizer=tokenizer)
    raise ValueError(f"unsupported merger: {name}")
