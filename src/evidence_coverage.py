"""Exact and contiguous-partial evidence matching against final context text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

TEXT_EVIDENCE_METRIC_MODE = "text_evidence_exact"


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        return payload
    if raw[0] == "{":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return records
            return [payload]

    records = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_no} must contain a JSON object")
        records.append(record)
    return records


def load_text_evidence_labels(
    path: Path, key_field: str = "id"
) -> dict[Hashable, dict[str, Any]]:
    """Load text-evidence labels keyed by query text or a stable record ID."""

    if key_field not in {"query", "id"}:
        raise ValueError(
            f"unsupported evidence label key {key_field!r}; expected 'query' or 'id'"
        )

    labels_by_key: dict[Hashable, dict[str, Any]] = {}
    for record in _load_json_records(path):
        query = record.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError(f"{path} contains an evidence label without a query")
        key = record.get(key_field)
        if key_field == "id" and (
            isinstance(key, bool) or not isinstance(key, (int, str))
        ):
            raise ValueError(f"{path} contains an invalid evidence label ID: {key!r}")
        if key in labels_by_key:
            raise ValueError(
                f"{path} contains a duplicate evidence label {key_field}: {key!r}"
            )

        evidence_texts = record.get("evidence_texts")
        evidence_passage_ids = record.get("evidence_passage_ids")
        if not isinstance(evidence_texts, list) or not all(
            isinstance(text, str) and text for text in evidence_texts
        ):
            raise ValueError(f"invalid evidence_texts for query {query!r}")
        if not isinstance(evidence_passage_ids, list) or not all(
            isinstance(passage_id, str) and passage_id
            for passage_id in evidence_passage_ids
        ):
            raise ValueError(f"invalid evidence_passage_ids for query {query!r}")
        if len(evidence_texts) != len(evidence_passage_ids):
            raise ValueError(f"evidence label lengths differ for query {query!r}")

        labels_by_key[key] = record
    return labels_by_key


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _normalized_characters(text: str) -> list[str]:
    """Normalize whitespace while preserving every non-whitespace character."""

    return list(_normalized_text(text))


def _token_ids(tokenizer, text: str) -> list[int]:
    normalized = _normalized_text(text)
    return [
        int(token_id)
        for token_id in tokenizer.encode(f" {normalized}", add_special_tokens=False)
    ]


def _longest_common_substring_length(
    pattern: Sequence[Hashable], text: Sequence[Hashable]
) -> int:
    """Return the longest exact contiguous substring shared by two sequences."""

    if not pattern or not text:
        return 0

    transitions: list[dict[Hashable, int]] = [{}]
    suffix_links = [-1]
    state_lengths = [0]
    last_state = 0

    for item in pattern:
        current_state = len(transitions)
        transitions.append({})
        suffix_links.append(0)
        state_lengths.append(state_lengths[last_state] + 1)

        previous_state = last_state
        while previous_state >= 0 and item not in transitions[previous_state]:
            transitions[previous_state][item] = current_state
            previous_state = suffix_links[previous_state]

        if previous_state < 0:
            suffix_links[current_state] = 0
        else:
            next_state = transitions[previous_state][item]
            if state_lengths[previous_state] + 1 == state_lengths[next_state]:
                suffix_links[current_state] = next_state
            else:
                clone_state = len(transitions)
                transitions.append(dict(transitions[next_state]))
                suffix_links.append(suffix_links[next_state])
                state_lengths.append(state_lengths[previous_state] + 1)
                while (
                    previous_state >= 0
                    and transitions[previous_state].get(item) == next_state
                ):
                    transitions[previous_state][item] = clone_state
                    previous_state = suffix_links[previous_state]
                suffix_links[next_state] = clone_state
                suffix_links[current_state] = clone_state
        last_state = current_state

    state = 0
    matched_length = 0
    longest = 0
    for item in text:
        while state and item not in transitions[state]:
            state = suffix_links[state]
            matched_length = state_lengths[state]
        next_state = transitions[state].get(item)
        if next_state is None:
            state = 0
            matched_length = 0
            continue
        state = next_state
        matched_length += 1
        longest = max(longest, matched_length)
        if longest == len(pattern):
            return longest
    return longest


def _partial_retention(compressed_overlap: int, retrieved_overlap: int):
    if retrieved_overlap == 0:
        return None
    return min(compressed_overlap, retrieved_overlap) / retrieved_overlap


def _exact_retention(compressed_exact: float, retrieved_exact: float):
    if not retrieved_exact:
        return None
    return compressed_exact


class TextEvidenceCoverageScorer:
    """Measure exact containment and longest contiguous partial evidence."""

    def __init__(self, metric_tokenizer):
        self.metric_tokenizer = metric_tokenizer

    def score(
        self,
        label: dict[str, Any],
        retrieved_context: str,
        compressed_context: str,
    ) -> dict[str, Any]:
        """Score one query using only context strings and gold evidence strings."""

        retrieved_chars = _normalized_characters(retrieved_context)
        compressed_chars = _normalized_characters(compressed_context)
        retrieved_tokens = _token_ids(self.metric_tokenizer, retrieved_context)
        compressed_tokens = _token_ids(self.metric_tokenizer, compressed_context)

        passage_details = []
        for passage_id, evidence_text in zip(
            label["evidence_passage_ids"], label["evidence_texts"]
        ):
            gold_chars = _normalized_characters(evidence_text)
            gold_tokens = _token_ids(self.metric_tokenizer, evidence_text)
            if not gold_chars:
                raise ValueError(
                    f"evidence passage {passage_id!r} has no normalized characters"
                )
            if not gold_tokens:
                raise ValueError(
                    f"evidence passage {passage_id!r} has no metric tokens"
                )

            retrieved_char_overlap = _longest_common_substring_length(
                gold_chars, retrieved_chars
            )
            compressed_char_overlap = _longest_common_substring_length(
                gold_chars, compressed_chars
            )
            retrieved_token_overlap = _longest_common_substring_length(
                gold_tokens, retrieved_tokens
            )
            compressed_token_overlap = _longest_common_substring_length(
                gold_tokens, compressed_tokens
            )
            retrieved_char_exact = float(retrieved_char_overlap == len(gold_chars))
            compressed_char_exact = float(compressed_char_overlap == len(gold_chars))
            retrieved_token_exact = float(retrieved_token_overlap == len(gold_tokens))
            compressed_token_exact = float(compressed_token_overlap == len(gold_tokens))
            passage_details.append(
                {
                    "passage_id": passage_id,
                    "gold_characters": len(gold_chars),
                    "gold_tokens": len(gold_tokens),
                    "retrieval_char_exact": retrieved_char_exact,
                    "retrieval_token_exact": retrieved_token_exact,
                    "compressed_char_exact": compressed_char_exact,
                    "compressed_token_exact": compressed_token_exact,
                    "retrieval_char_contiguous_partial_recall": retrieved_char_overlap
                    / len(gold_chars),
                    "retrieval_token_contiguous_partial_recall": retrieved_token_overlap
                    / len(gold_tokens),
                    "compressed_char_contiguous_partial_recall": compressed_char_overlap
                    / len(gold_chars),
                    "compressed_token_contiguous_partial_recall": compressed_token_overlap
                    / len(gold_tokens),
                    "conditional_char_exact_retention": _exact_retention(
                        compressed_char_exact, retrieved_char_exact
                    ),
                    "conditional_token_exact_retention": _exact_retention(
                        compressed_token_exact, retrieved_token_exact
                    ),
                    "conditional_char_contiguous_partial_retention": _partial_retention(
                        compressed_char_overlap, retrieved_char_overlap
                    ),
                    "conditional_token_contiguous_partial_retention": _partial_retention(
                        compressed_token_overlap, retrieved_token_overlap
                    ),
                }
            )

        return {
            "gold": {
                "passage_count": len(passage_details),
                "characters": sum(
                    detail["gold_characters"] for detail in passage_details
                ),
                "tokens": sum(detail["gold_tokens"] for detail in passage_details),
            },
            "retrieval": _summarize_passage_matches(passage_details, "retrieval"),
            "compressed": _summarize_passage_matches(passage_details, "compressed"),
            "conditional": _summarize_conditional_matches(passage_details),
            "passages": passage_details,
        }


def _summarize_passage_matches(
    details: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    char_exact = [detail[f"{prefix}_char_exact"] for detail in details]
    token_exact = [detail[f"{prefix}_token_exact"] for detail in details]
    char_partial = [
        detail[f"{prefix}_char_contiguous_partial_recall"] for detail in details
    ]
    token_partial = [
        detail[f"{prefix}_token_contiguous_partial_recall"] for detail in details
    ]
    return {
        "char_exact_recall": mean(char_exact),
        "token_exact_recall": mean(token_exact),
        "any_char_exact": float(any(char_exact)),
        "any_token_exact": float(any(token_exact)),
        "all_char_exact": float(all(char_exact)),
        "all_token_exact": float(all(token_exact)),
        "char_contiguous_partial_recall": mean(char_partial),
        "token_contiguous_partial_recall": mean(token_partial),
    }


def _summarize_conditional_matches(
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    def values(name: str):
        return [detail[name] for detail in details if detail[name] is not None]

    char_exact = values("conditional_char_exact_retention")
    token_exact = values("conditional_token_exact_retention")
    char_partial = values("conditional_char_contiguous_partial_retention")
    token_partial = values("conditional_token_contiguous_partial_retention")
    return {
        "char_exact_retention": _optional_mean(char_exact),
        "token_exact_retention": _optional_mean(token_exact),
        "char_exact_eligible_passage_count": len(char_exact),
        "token_exact_eligible_passage_count": len(token_exact),
        "char_contiguous_partial_retention": _optional_mean(char_partial),
        "token_contiguous_partial_retention": _optional_mean(token_partial),
        "contiguous_partial_eligible_passage_count": len(char_partial),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _optional_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_text_evidence_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
        "metric_mode": TEXT_EVIDENCE_METRIC_MODE,
        "primary_metric": "char_exact_recall",
        "gold": {
            "characters": mean(record["gold"]["characters"] for record in records),
            "tokens": mean(record["gold"]["tokens"] for record in records),
            "passage_count": mean(
                record["gold"]["passage_count"] for record in records
            ),
        },
    }
    section_keys = (
        "char_exact_recall",
        "token_exact_recall",
        "any_char_exact",
        "any_token_exact",
        "all_char_exact",
        "all_token_exact",
        "char_contiguous_partial_recall",
        "token_contiguous_partial_recall",
        "context_tokens",
    )
    for section in ("retrieval", "compressed"):
        summary[section] = {
            key: mean(record[section][key] for record in records)
            for key in section_keys
        }

    conditional_keys = (
        "char_exact_retention",
        "token_exact_retention",
        "char_contiguous_partial_retention",
        "token_contiguous_partial_retention",
    )
    summary["conditional"] = {
        key: _optional_mean(
            record["conditional"][key]
            for record in records
            if record["conditional"][key] is not None
        )
        for key in conditional_keys
    }
    summary["conditional"].update(
        {
            "char_exact_defined_query_count": sum(
                record["conditional"]["char_exact_retention"] is not None
                for record in records
            ),
            "token_exact_defined_query_count": sum(
                record["conditional"]["token_exact_retention"] is not None
                for record in records
            ),
            "contiguous_partial_defined_query_count": sum(
                record["conditional"]["char_contiguous_partial_retention"] is not None
                for record in records
            ),
        }
    )
    return summary
