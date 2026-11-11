#!/usr/bin/env python
"""Prepare NewsQA QA files using the official single consensus answer."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CSV_MEMBER = "newsqa-data-v1/newsqa-data-v1.csv"
EXPECTED_RAW_ANSWERABLE_TEST_ROWS = 5_126
EXPECTED_CONSENSUS_TEST_ROWS = 4_293


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_test_story_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        story_ids = {row["story_id"] for row in csv.DictReader(handle)}
    if not story_ids:
        raise ValueError(f"no test story IDs in {path}")
    return story_ids


def load_raw_answerable_test_rows(
    annotations_zip: Path, test_story_ids: set[str]
) -> list[dict[str, str]]:
    with zipfile.ZipFile(annotations_zip) as archive:
        with archive.open(CSV_MEMBER) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
            rows = [
                row
                for row in csv.DictReader(text_handle)
                if row["story_id"] in test_story_ids
                and any(
                    answer != "None" for answer in row["answer_char_ranges"].split("|")
                )
            ]
    if len(rows) != EXPECTED_RAW_ANSWERABLE_TEST_ROWS:
        raise ValueError(
            f"expected {EXPECTED_RAW_ANSWERABLE_TEST_ROWS} raw-answerable test rows, "
            f"got {len(rows)}"
        )
    return rows


def _clip_range(raw_range: str, story_length: int) -> str | None:
    if ":" not in raw_range:
        return raw_range
    start, end = map(int, raw_range.split(":"))
    end = min(end, story_length)
    if start >= end:
        return None
    return f"{start}:{end}"


def _official_clipped_annotations(
    row: dict[str, str], story_length: int
) -> tuple[str, str]:
    """Port the range clipping performed by ``NewsQaDataset.__init__``."""
    clipped_users = []
    for user_answers in row["answer_char_ranges"].split("|"):
        clipped_answers = [
            clipped
            for raw_range in user_answers.split(",")
            if (clipped := _clip_range(raw_range, story_length)) is not None
        ]
        if clipped_answers:
            clipped_users.append(",".join(clipped_answers))

    clipped_validated = ""
    if row["validated_answers"]:
        validated_answers = json.loads(row["validated_answers"])
        clipped_validated_answers = {}
        for raw_range, count in validated_answers.items():
            clipped = _clip_range(raw_range, story_length)
            if clipped is not None:
                clipped_validated_answers[clipped] = count
        clipped_validated = json.dumps(
            clipped_validated_answers,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return "|".join(clipped_users), clipped_validated


def official_consensus_span(
    row: dict[str, str], story_length: int
) -> tuple[int, int] | None:
    """Match the official range clipping and ``get_consensus_answer``."""
    answer_char_ranges, validated_answers_json = _official_clipped_annotations(
        row, story_length
    )
    if validated_answers_json:
        validated_answers = json.loads(validated_answers_json)
        answer, max_count = max(validated_answers.items(), key=lambda item: item[1])
        total_count = sum(validated_answers.values())
        if max_count >= total_count / 2.0 and answer not in {
            "none",
            "bad_question",
        }:
            return tuple(map(int, answer.split(":")))
        return None

    answers: Counter[str] = Counter()
    for user_answer in answer_char_ranges.split("|"):
        for answer in user_answer.split(","):
            answers[answer] += 1
    top_answer = answers.most_common(1)
    if top_answer and ":" in top_answer[0][0]:
        return tuple(map(int, top_answer[0][0].split(":")))
    return None


def build_consensus_records(
    rows: list[dict[str, str]],
    raw_questions: list[dict[str, Any]],
    documents_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != len(raw_questions):
        raise ValueError(
            f"official-row/question count mismatch: {len(rows)} != {len(raw_questions)}"
        )

    questions = []
    answers = []
    for local_id, (row, raw_question) in enumerate(zip(rows, raw_questions)):
        if raw_question.get("id") != local_id:
            raise ValueError(f"unexpected raw question ID at row {local_id}")
        if raw_question.get("query") != row["question"]:
            raise ValueError(f"official-row/question text mismatch at row {local_id}")

        story_sha1 = Path(row["story_id"]).stem
        story_text = (documents_dir / f"doc_{story_sha1}.txt").read_text(
            encoding="utf-8"
        )
        span = official_consensus_span(row, len(story_text))
        if span is None:
            continue

        start, end = span
        if not 0 <= start < end <= len(story_text):
            raise ValueError(
                f"invalid consensus span at row {local_id}: {(start, end)}"
            )
        answer = story_text[start:end]
        if not answer:
            raise ValueError(f"empty consensus answer at row {local_id}")

        questions.append(dict(raw_question))
        answers.append({"id": local_id, "answers": [answer]})

    if len(questions) != EXPECTED_CONSENSUS_TEST_ROWS:
        raise ValueError(
            f"expected {EXPECTED_CONSENSUS_TEST_ROWS} consensus test rows, "
            f"got {len(questions)}"
        )
    return questions, answers
