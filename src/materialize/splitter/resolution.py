from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

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
PRONOUN_CANDIDATES = PRONOUN_SUBJECTS | PRONOUN_POSSESSIVES
LEADING_TOKEN_RE = re.compile(r"^(\s*)([A-Za-z][A-Za-z'’-]*)(.*)$")
WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
CASE_TITLE_RE = re.compile(
    r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*\s+v\.\s+[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*)\b"
)
SINGLE_PROPER_RE = re.compile(r"\b([A-Z][\w.-]*)\b")
PROPER_NAME_RE = re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){1,5})\b")
ARTICLE_NP_RE = re.compile(
    r"\b((?:the|a|an|this|that|these|those)\s+[A-Za-z][\w.-]*(?:\s+[A-Za-z][\w.-]*){0,6})\b",
    re.IGNORECASE,
)
LIKELY_TRAILING_VERBS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "became",
    "become",
    "bring",
    "brought",
    "called",
    "decided",
    "had",
    "has",
    "have",
    "included",
    "include",
    "includes",
    "is",
    "made",
    "make",
    "makes",
    "redefined",
    "referred",
    "said",
    "was",
    "were",
    "won",
}

OPENAI_RESOLUTION_TIMEOUT_SEC = 600
OPENAI_RESOLUTION_MAX_RETRIES = 6
OPENAI_RESOLUTION_FALLBACK_WINDOW_SENTENCES = 40
OPENAI_RESOLUTION_RETRYABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


@dataclass(frozen=True)
class CandidateSpan:
    index: int
    text: str
    sentence_index: int
    source: str


@dataclass(frozen=True)
class PronounTarget:
    token: str
    char_start: int
    char_end: int
    word_index: int


def word_tokenize(sentence: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", sentence)


def detokenize(words: Sequence[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    return text


def strip_trailing_punctuation(tokens: Sequence[str]) -> List[str]:
    cleaned = list(tokens)
    while cleaned and cleaned[-1] in {".", ",", ";", ":", "!", "?"}:
        cleaned.pop()
    return cleaned


def looks_like_trailing_verb(token: str) -> bool:
    lowered = token.lower()
    if lowered in LIKELY_TRAILING_VERBS:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z'’-]*", token):
        if lowered.endswith("ed") or lowered.endswith("ing"):
            return True
    return False


def strip_appositive_tail(text: str) -> str:
    stripped = text.strip()
    if "(" in stripped and stripped.endswith(")"):
        base = re.sub(r"\s+\([^)]*\)$", "", stripped).strip()
        if base and PROPER_NAME_RE.fullmatch(base):
            return base
    return stripped


def clean_antecedent_tokens(tokens: Sequence[str]) -> List[str]:
    cleaned = strip_trailing_punctuation(tokens)
    if not cleaned:
        return []

    while cleaned and looks_like_trailing_verb(cleaned[-1]):
        candidate = cleaned[:-1]
        candidate = strip_trailing_punctuation(candidate)
        if not candidate:
            break
        cleaned = candidate

    return cleaned


def build_sentence_word_index(
    sentences: Sequence[str],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    words: List[str] = []
    sentence_word_spans: List[Tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        sent_words = word_tokenize(sentence)
        start = cursor
        words.extend(sent_words)
        cursor += len(sent_words)
        sentence_word_spans.append((start, cursor))
    return words, sentence_word_spans


def normalize_clusters(prediction: Any) -> List[List[Tuple[int, int]]]:
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


def run_fastcoref(words: Sequence[str], model) -> Any:
    predictions = model.predict(texts=[list(words)], is_split_into_words=True)
    if not predictions:
        raise ValueError("fastcoref returned no predictions")
    return predictions[0]


def find_leading_pronoun_cluster(
    sent_start: int,
    sent_end: int,
    words: Sequence[str],
    mention_to_cluster: dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[Tuple[int, int]] | None:
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


def build_possessive(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.endswith("s"):
        return f"{text}'"
    return f"{text}'s"


def clean_antecedent_text_from_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    cleaned_tokens = clean_antecedent_tokens(tokens)
    cleaned_text = (
        detokenize(cleaned_tokens).strip()
        if cleaned_tokens
        else detokenize(tokens).strip()
    )
    return strip_appositive_tail(cleaned_text).strip()


def find_pronoun_target(sentence: str) -> PronounTarget | None:
    for word_index, match in enumerate(WORD_TOKEN_RE.finditer(sentence)):
        token = match.group(0)
        if token.lower() in PRONOUN_CANDIDATES:
            return PronounTarget(
                token=token,
                char_start=match.start(),
                char_end=match.end(),
                word_index=word_index,
            )
    return None


def collect_candidate_texts(sentence: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for pattern, source in (
        (CASE_TITLE_RE, "case_title"),
        (PROPER_NAME_RE, "proper_name"),
        (SINGLE_PROPER_RE, "single_proper_name"),
        (ARTICLE_NP_RE, "article_np"),
    ):
        for match in pattern.finditer(sentence):
            text = match.group(1).strip()
            if text:
                candidates.append((text, source))
    return candidates


def build_candidate_spans(sentences: Sequence[str]) -> List[CandidateSpan]:
    candidates: List[CandidateSpan] = []
    seen: set[Tuple[str, int]] = set()
    for sentence_index, sentence in enumerate(sentences):
        for raw_text, source in collect_candidate_texts(sentence):
            cleaned_text = clean_antecedent_text_from_tokens(word_tokenize(raw_text))
            if not cleaned_text:
                continue
            key = (cleaned_text, sentence_index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateSpan(
                    index=len(candidates),
                    text=cleaned_text,
                    sentence_index=sentence_index,
                    source=source,
                )
            )
    return candidates


def apply_target_replacement(
    sentence: str, target: PronounTarget, replacement: str
) -> str:
    return f"{sentence[:target.char_start]}{replacement}{sentence[target.char_end:]}"


def is_demonstrative_np_target(sentence: str, target: PronounTarget) -> bool:
    lowered = target.token.lower()
    if lowered not in {"this", "that", "these", "those"}:
        return False
    following = sentence[target.char_end :]
    return WORD_TOKEN_RE.search(following) is not None


def build_openai_client(project_root: Path | str, api_key: str | None = None) -> OpenAI:
    if api_key is None:
        load_dotenv(Path(project_root) / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing from .env")
    return OpenAI(api_key=api_key)


def build_openai_resolution_prompt(
    sentences: Sequence[str],
    candidate_spans: Sequence[CandidateSpan],
    targets: Sequence[dict[str, Any]],
):
    return {
        "instructions": [
            "Resolve pronouns using the full document context.",
            "For each target sentence, choose at most one antecedent from candidate_spans.",
            "You must select only from candidate_spans. Do not invent text.",
            "Only resolve pronouns that need cross-sentence context. If the reference can be understood within the same sentence, prefer no replacement.",
            "Use only candidates from earlier sentences. A noun phrase from the same sentence should never be selected as an antecedent.",
            "If the pronoun can only be linked to something inside the same sentence, do not resolve it at all.",
            "Choose the smallest noun phrase or named entity that makes the substituted sentence semantically coherent.",
            "Do not choose a phrase containing a trailing predicate or event verb.",
            "For possessives like 'His', choose the entity mention and the code will add the possessive suffix.",
            "In legal or case-style text, 'It' often refers to the case, decision, or standard rather than the court.",
            "Be highly conservative overall. If there is any doubt, prefer no replacement.",
            "Do not resolve demonstrative noun phrases such as 'this case', 'that decision', 'this event', 'this fact', 'this film', or similar 'this/that/these/those + noun' forms.",
            "If the target is a demonstrative followed by a common noun, prefer no replacement even if an earlier named entity exists.",
            "Prefer no replacement if no candidate yields a clearly coherent sentence.",
            "Before you return a replacement, verify that the selected antecedent appears in an earlier sentence, not the same sentence.",
            "Do not return candidate_index.",
            "Return only candidate_text copied verbatim from candidate_spans and candidate_sentence_index.",
            "If candidate_text and candidate_sentence_index do not exactly match one provided candidate span, the answer will be rejected.",
            "Return valid JSON only with key 'resolutions'.",
            "Each item must contain: sentence_index, replace, candidate_text, candidate_sentence_index.",
            "candidate_text and candidate_sentence_index must be null when replace is false.",
        ],
        "sentences": list(sentences),
        "candidate_spans": [asdict(candidate) for candidate in candidate_spans],
        "targets": list(targets),
    }


def is_context_length_exceeded_error(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == "context_length_exceeded":
            return True
    return "context length" in str(exc).lower()


def parse_openai_json_object(content: str):
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    first_brace = stripped.find("{")
    if first_brace < 0:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    obj, _ = decoder.raw_decode(stripped[first_brace:])
    return obj


def request_openai_resolution_payload(
    client: OpenAI, model: str, prompt: dict[str, Any]
):
    last_error = None
    for attempt in range(OPENAI_RESOLUTION_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=OPENAI_RESOLUTION_TIMEOUT_SEC,
            )
            content = response.choices[0].message.content or ""
            return parse_openai_json_object(content)
        except OPENAI_RESOLUTION_RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == OPENAI_RESOLUTION_MAX_RETRIES - 1:
                raise
            sleep_sec = min(60, 2**attempt)
            time.sleep(sleep_sec)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == OPENAI_RESOLUTION_MAX_RETRIES - 1:
                raise ValueError(
                    "OpenAI antecedent selection returned invalid JSON after retries"
                ) from exc
            sleep_sec = min(60, 2**attempt)
            time.sleep(sleep_sec)
    raise (
        last_error
        if last_error is not None
        else RuntimeError(
            "OpenAI resolution request failed without a captured exception"
        )
    )


def resolve_pronouns_with_openai_windowed(
    sentences: Sequence[str],
    targets: Sequence[PronounTarget | None],
    candidate_spans: Sequence[CandidateSpan],
    client: OpenAI,
    model: str,
):
    resolutions = []
    for sentence_index, target in enumerate(targets):
        if target is None:
            continue
        window_size = OPENAI_RESOLUTION_FALLBACK_WINDOW_SENTENCES
        while True:
            start_idx = max(0, sentence_index - window_size)
            local_sentences = list(sentences[start_idx : sentence_index + 1])
            local_candidates = []
            for candidate in candidate_spans:
                if start_idx <= candidate.sentence_index < sentence_index:
                    local_candidates.append(
                        CandidateSpan(
                            index=len(local_candidates),
                            text=candidate.text,
                            sentence_index=candidate.sentence_index - start_idx,
                            source=candidate.source,
                        )
                    )

            if not local_candidates:
                resolutions.append(
                    {
                        "sentence_index": sentence_index,
                        "replace": False,
                        "candidate_text": None,
                        "candidate_sentence_index": None,
                    }
                )
                break

            local_target = {
                "sentence_index": sentence_index - start_idx,
                "sentence": local_sentences[-1],
                "target_token": target.token,
                "target_word_index": target.word_index,
            }
            prompt = build_openai_resolution_prompt(
                local_sentences,
                local_candidates,
                [local_target],
            )
            try:
                parsed = request_openai_resolution_payload(client, model, prompt)
                local_resolutions = parsed.get("resolutions")
                if not isinstance(local_resolutions, list) or not local_resolutions:
                    resolutions.append(
                        {
                            "sentence_index": sentence_index,
                            "replace": False,
                            "candidate_text": None,
                            "candidate_sentence_index": None,
                        }
                    )
                    break
                local_resolution = local_resolutions[0]
                global_resolution = dict(local_resolution)
                global_resolution["sentence_index"] = sentence_index
                local_candidate_sentence_index = global_resolution.get(
                    "candidate_sentence_index"
                )
                if local_candidate_sentence_index is not None:
                    global_resolution["candidate_sentence_index"] = (
                        local_candidate_sentence_index + start_idx
                    )
                resolutions.append(global_resolution)
                break
            except BadRequestError as exc:
                if not is_context_length_exceeded_error(exc):
                    raise
                if window_size <= 4:
                    resolutions.append(
                        {
                            "sentence_index": sentence_index,
                            "replace": False,
                            "candidate_text": None,
                            "candidate_sentence_index": None,
                        }
                    )
                    break
                window_size = max(4, window_size // 2)
    return resolutions


def resolve_leading_pronouns_with_fastcoref(sentences: Sequence[str], coref_model):
    if not sentences:
        return list(sentences), []

    words, sentence_word_spans = build_sentence_word_index(sentences)
    if not words:
        return list(sentences), []

    prediction = run_fastcoref(words, coref_model)
    clusters = normalize_clusters(prediction)
    mention_to_cluster: dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for cluster in clusters:
        cluster = sorted(cluster)
        for mention in cluster:
            mention_to_cluster[mention] = cluster

    rewritten = list(sentences)
    records = []
    for sentence_index, sentence in enumerate(sentences):
        match = LEADING_TOKEN_RE.match(sentence)
        if not match:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "no_leading_token_match",
                }
            )
            continue
        prefix, token, suffix = match.groups()
        lowered = token.lower()
        if lowered not in PRONOUN_SUBJECTS and lowered not in PRONOUN_POSSESSIVES:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "leading_token_not_pronoun",
                }
            )
            continue

        sent_start, sent_end = sentence_word_spans[sentence_index]
        cluster = find_leading_pronoun_cluster(
            sent_start, sent_end, words, mention_to_cluster
        )
        if not cluster:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "no_coref_cluster",
                }
            )
            continue

        antecedent_words: list[str] | None = None
        antecedent_span: Tuple[int, int] | None = None
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
            antecedent_words = mention_tokens
            antecedent_span = (mention_start, mention_end)

        if not antecedent_words:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "no_prior_non_pronominal_mention",
                }
            )
            continue

        raw_antecedent_text = detokenize(antecedent_words).strip()
        antecedent_text = (
            clean_antecedent_text_from_tokens(antecedent_words) or raw_antecedent_text
        )
        if lowered in PRONOUN_POSSESSIVES:
            replacement = build_possessive(antecedent_text)
        else:
            replacement = antecedent_text
        resolved = f"{prefix}{replacement}{suffix}"
        rewritten[sentence_index] = resolved
        records.append(
            {
                "sentence_index": sentence_index,
                "original": sentence,
                "resolved": resolved,
                "changed": resolved != sentence,
                "reason": "resolved_with_fastcoref",
                "replacement": replacement,
                "raw_antecedent_text": raw_antecedent_text,
                "antecedent_span": (
                    list(antecedent_span) if antecedent_span is not None else None
                ),
                "cluster": [list(item) for item in cluster],
            }
        )

    return rewritten, records


def resolve_pronouns_with_openai(sentences: Sequence[str], client: OpenAI, model: str):
    candidate_spans = build_candidate_spans(sentences)
    if not candidate_spans:
        return list(sentences), []

    targets = []
    for sentence_index, sentence in enumerate(sentences):
        target = find_pronoun_target(sentence)
        if target is None:
            targets.append(None)
        else:
            targets.append(target)

    full_targets = [
        {
            "sentence_index": idx,
            "sentence": sentences[idx],
            "target_token": target.token,
            "target_word_index": target.word_index,
        }
        for idx, target in enumerate(targets)
        if target is not None
    ]
    prompt = build_openai_resolution_prompt(sentences, candidate_spans, full_targets)
    try:
        parsed = request_openai_resolution_payload(client, model, prompt)
        resolutions = parsed.get("resolutions")
        if not isinstance(resolutions, list):
            raise ValueError(
                "OpenAI antecedent selection did not return a 'resolutions' list"
            )
    except BadRequestError as exc:
        if not is_context_length_exceeded_error(exc):
            raise
        resolutions = resolve_pronouns_with_openai_windowed(
            sentences=sentences,
            targets=targets,
            candidate_spans=candidate_spans,
            client=client,
            model=model,
        )

    by_sentence = {}
    for item in resolutions:
        if isinstance(item, dict) and "sentence_index" in item:
            by_sentence[int(item["sentence_index"])] = item

    rewritten = list(sentences)
    records = []
    for sentence_index, sentence in enumerate(sentences):
        target = targets[sentence_index]
        if target is None:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "no_pronoun_target",
                }
            )
            continue

        selection = by_sentence.get(sentence_index)
        if not selection or not bool(selection.get("replace", False)):
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "openai_no_replacement",
                    "target_token": target.token,
                }
            )
            continue

        if is_demonstrative_np_target(sentence, target):
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "openai_demonstrative_np_rejected",
                    "target_token": target.token,
                }
            )
            continue

        selected_text = selection.get("candidate_text")
        selected_sentence_index = selection.get("candidate_sentence_index")
        matching_candidates = [
            candidate
            for candidate in candidate_spans
            if candidate.text == selected_text
            and candidate.sentence_index == selected_sentence_index
        ]
        if len(matching_candidates) != 1:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "openai_invalid_candidate_selection",
                    "target_token": target.token,
                    "candidate_text": selected_text,
                    "candidate_sentence_index": selected_sentence_index,
                }
            )
            continue
        candidate = matching_candidates[0]

        if candidate.sentence_index >= sentence_index:
            records.append(
                {
                    "sentence_index": sentence_index,
                    "original": sentence,
                    "resolved": sentence,
                    "changed": False,
                    "reason": "openai_same_sentence_or_future_candidate_rejected",
                    "target_token": target.token,
                    "candidate_text": candidate.text,
                    "candidate_sentence_index": candidate.sentence_index,
                }
            )
            continue

        replacement = (
            build_possessive(candidate.text)
            if target.token.lower() in PRONOUN_POSSESSIVES
            else candidate.text
        )
        resolved = apply_target_replacement(sentence, target, replacement)
        rewritten[sentence_index] = resolved
        records.append(
            {
                "sentence_index": sentence_index,
                "original": sentence,
                "resolved": resolved,
                "changed": resolved != sentence,
                "reason": "resolved_with_openai",
                "target_token": target.token,
                "replacement": replacement,
                "candidate_index": candidate.index,
                "candidate_text": candidate.text,
                "candidate_sentence_index": candidate.sentence_index,
                "candidate_source": candidate.source,
            }
        )

    return rewritten, records
