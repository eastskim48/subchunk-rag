#!/usr/bin/env python
"""Build runnable QA and evidence files for the custom HotpotQA corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

LINK_TAG = re.compile(r"</?a(?:\s+[^>]*)?>", re.IGNORECASE)


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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_all(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    spans = []
    start = 0
    while True:
        position = text.find(needle, start)
        if position < 0:
            return spans
        spans.append((position, position + len(needle)))
        start = position + 1


def normalize_whitespace_with_char_map(text: str) -> tuple[str, list[int]]:
    normalized_characters = []
    source_indices = []
    pending_space_index = None
    for index, character in enumerate(text):
        if character.isspace():
            if normalized_characters and pending_space_index is None:
                pending_space_index = index
            continue
        if pending_space_index is not None:
            normalized_characters.append(" ")
            source_indices.append(pending_space_index)
            pending_space_index = None
        normalized_characters.append(character)
        source_indices.append(index)
    return "".join(normalized_characters), source_indices


def locate_whitespace_normalized_spans(
    document_text: str, evidence_text: str
) -> list[tuple[int, int]]:
    normalized_document, source_indices = normalize_whitespace_with_char_map(
        document_text
    )
    normalized_evidence, _ = normalize_whitespace_with_char_map(evidence_text)
    normalized_spans = find_all(normalized_document, normalized_evidence)
    spans = []
    for start, end in normalized_spans:
        source_start = source_indices[start]
        source_end = source_indices[end - 1] + 1
        actual_text = document_text[source_start:source_end]
        actual_normalized, _ = normalize_whitespace_with_char_map(actual_text)
        if actual_normalized != normalized_evidence:
            raise ValueError("whitespace span reverse mapping failed")
        spans.append((source_start, source_end))
    return spans


def locate_evidence_span(
    *,
    document_text: str,
    comparison_sentence_text: str,
    validation_status: str,
) -> tuple[int, int, str]:
    if validation_status == "trimmed_exact_match":
        spans = find_all(document_text, comparison_sentence_text.strip())
        match_mode = "trimmed_exact"
    elif validation_status == "whitespace_normalized_match_only":
        spans = locate_whitespace_normalized_spans(
            document_text, comparison_sentence_text
        )
        match_mode = "whitespace_normalized"
    else:
        raise ValueError(f"cannot locate evidence for status {validation_status!r}")
    if len(spans) != 1:
        raise ValueError(
            f"expected exactly one {match_mode} evidence span, got {len(spans)}"
        )
    start, end = spans[0]
    return start, end, match_mode


def source_sentence_plaintext(text: str) -> str:
    return html.unescape(LINK_TAG.sub("", text))


def page_plaintext_and_sentence_spans(
    page: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    document_parts = []
    paragraph_layouts = []
    document_length = 0
    for source_sentences in page["text"]:
        cleaned_sentences = [
            source_sentence_plaintext(sentence) for sentence in source_sentences
        ]
        joined = "".join(cleaned_sentences)
        left = len(joined) - len(joined.lstrip())
        right = len(joined.rstrip())
        paragraph = joined[left:right]
        if not paragraph:
            continue
        if document_parts:
            document_parts.append("\n\n")
            document_length += 2
        paragraph_start = document_length
        document_parts.append(paragraph)
        document_length += len(paragraph)

        sentence_spans = []
        source_offset = 0
        for sentence in cleaned_sentences:
            source_start = source_offset
            source_end = source_start + len(sentence)
            source_offset = source_end
            clipped_start = max(source_start, left)
            clipped_end = min(source_end, right)
            if clipped_start >= clipped_end:
                sentence_spans.append([paragraph_start, paragraph_start])
                continue
            start = paragraph_start + clipped_start - left
            end = paragraph_start + clipped_end - left
            while start < end and paragraph[start - paragraph_start].isspace():
                start += 1
            while end > start and paragraph[end - paragraph_start - 1].isspace():
                end -= 1
            sentence_spans.append([start, end])
        paragraph_layouts.append(
            {"sentences": cleaned_sentences, "sentence_spans": sentence_spans}
        )
    return "".join(document_parts).strip() + "\n", paragraph_layouts


def _sentence_matches(source: str, target: str, validation_status: str) -> bool:
    if validation_status == "trimmed_exact_match":
        return source.strip() == target.strip()
    if validation_status == "whitespace_normalized_match_only":
        return " ".join(source.split()) == " ".join(target.split())
    return False


def find_structural_evidence_span(
    *,
    document_text: str,
    raw_page: dict[str, Any],
    context_sentences: list[str],
    sentence_index: int,
    comparison_sentence_text: str,
    validation_status: str,
) -> tuple[int, int, str, str]:
    reconstructed_text, paragraph_layouts = page_plaintext_and_sentence_spans(raw_page)
    if reconstructed_text != document_text:
        raise ValueError(
            f"raw page {raw_page['id']} does not reconstruct its stored document"
        )
    context_plaintext = [
        source_sentence_plaintext(sentence) for sentence in context_sentences
    ]
    trimmed_context = [sentence.strip() for sentence in context_plaintext]
    normalized_context = [" ".join(sentence.split()) for sentence in context_plaintext]

    exact_candidates = [
        paragraph
        for paragraph in paragraph_layouts
        if [sentence.strip() for sentence in paragraph["sentences"]] == trimmed_context
    ]
    if len(exact_candidates) == 1:
        candidates = exact_candidates
        paragraph_match_mode = "source_paragraph_exact"
    else:
        candidates = [
            paragraph
            for paragraph in paragraph_layouts
            if [" ".join(sentence.split()) for sentence in paragraph["sentences"]]
            == normalized_context
        ]
        paragraph_match_mode = "source_paragraph_whitespace_normalized"

    if len(candidates) != 1:
        scored_candidates = []
        for paragraph in paragraph_layouts:
            source_sentences = paragraph["sentences"]
            if len(source_sentences) != len(context_plaintext):
                continue
            if not 0 <= sentence_index < len(source_sentences):
                continue
            if not _sentence_matches(
                source_sentences[sentence_index],
                comparison_sentence_text,
                validation_status,
            ):
                continue
            normalized_sentences = [
                " ".join(value.split()) for value in source_sentences
            ]
            score = sum(
                source == context
                for source, context in zip(normalized_sentences, normalized_context)
            )
            scored_candidates.append((score, paragraph))
        if not scored_candidates:
            raise ValueError("no source paragraph contains the validated evidence")
        best_score = max(score for score, _ in scored_candidates)
        candidates = [
            paragraph for score, paragraph in scored_candidates if score == best_score
        ]
        if len(candidates) != 1:
            raise ValueError("source paragraph alignment is not unique")
        paragraph_match_mode = "source_sentence_index_max_agreement"

    paragraph = candidates[0]
    if not 0 <= sentence_index < len(paragraph["sentence_spans"]):
        raise ValueError(f"invalid source sentence index {sentence_index}")
    start, end = paragraph["sentence_spans"][sentence_index]
    evidence_text = document_text[start:end]
    if validation_status == "trimmed_exact_match":
        if evidence_text != comparison_sentence_text.strip():
            raise ValueError("structural exact evidence differs from validated text")
        text_match_mode = "trimmed_exact"
    elif validation_status == "whitespace_normalized_match_only":
        if " ".join(evidence_text.split()) != " ".join(
            comparison_sentence_text.split()
        ):
            raise ValueError(
                "structural whitespace evidence differs from validated text"
            )
        text_match_mode = "whitespace_normalized"
    else:
        raise ValueError(f"unsupported validation status: {validation_status}")
    return start, end, text_match_mode, paragraph_match_mode


def load_selected_raw_pages(
    path: Path, page_ids: set[str]
) -> dict[str, dict[str, Any]]:
    pages = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            page = json.loads(line)
            page_id = str(page["id"])
            if page_id not in page_ids:
                continue
            if page_id in pages:
                raise ValueError(
                    f"duplicate raw page ID {page_id} at line {line_number}"
                )
            pages[page_id] = page
    missing = page_ids - set(pages)
    if missing:
        raise ValueError(f"missing selected raw pages: {sorted(missing)[:5]}")
    return pages


def build_records(
    *,
    dataset_root: Path,
    dev_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    title_to_documents: dict[str, list[dict[str, str]]],
    raw_pages_by_id: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    validation_by_source_id = {}
    for record in validation_records:
        source_id = record["_id"]
        if source_id in validation_by_source_id:
            raise ValueError(f"duplicate validation source ID: {source_id}")
        validation_by_source_id[source_id] = record

    @lru_cache(maxsize=None)
    def load_document(relative_path: str) -> str:
        return (dataset_root / relative_path).read_text(encoding="utf-8")

    questions = []
    answers = []
    evidence_records = []
    annotation_issues = []
    seen_questions = set()
    for local_id, source_record in enumerate(dev_records):
        source_id = source_record["_id"]
        question = source_record["question"]
        answer = source_record["answer"]
        if question in seen_questions:
            raise ValueError(f"duplicate HotpotQA question: {question!r}")
        seen_questions.add(question)
        validation = validation_by_source_id.get(source_id)
        if validation is None:
            raise ValueError(f"missing validation record for {source_id}")
        source_facts = source_record["supporting_facts"]
        validated_facts = validation["supporting_facts"]
        if len(source_facts) != len(validated_facts):
            raise ValueError(f"supporting-fact count mismatch for {source_id}")
        context_by_title = {
            title: sentences for title, sentences in source_record["context"]
        }

        questions.append(
            {
                "id": local_id,
                "query": question,
                "source_id": source_id,
                "level": source_record["level"],
                "type": source_record["type"],
            }
        )
        answers.append(
            {
                "id": local_id,
                "answers": [answer],
                "answer": answer,
                "source_id": source_id,
            }
        )

        evidence_passage_ids = []
        evidence_texts = []
        evidence_char_spans = []
        evidence_page_ids = []
        evidence_document_files = []
        evidence_titles = []
        supporting_fact_sentence_indices = []
        source_supporting_fact_texts = []
        evidence_span_match_modes = []
        evidence_alignment_modes = []
        invalid_supporting_facts = []

        for fact_index, (source_fact, validated_fact) in enumerate(
            zip(source_facts, validated_facts)
        ):
            title, sentence_index = source_fact
            if (
                validated_fact["title"] != title
                or validated_fact["sentence_index"] != sentence_index
            ):
                raise ValueError(f"supporting-fact alignment mismatch for {source_id}")
            status = validated_fact["status"]
            if status not in {
                "trimmed_exact_match",
                "whitespace_normalized_match_only",
            }:
                issue = {
                    "source_id": source_id,
                    "local_id": local_id,
                    "fact_index": fact_index,
                    "title": title,
                    "sentence_index": sentence_index,
                    "status": status,
                    "sentence_text": validated_fact["sentence_text"],
                    "comparison_sentence_text": validated_fact[
                        "comparison_sentence_text"
                    ],
                }
                invalid_supporting_facts.append(issue)
                annotation_issues.append(issue)
                continue

            matched_page_ids = (
                validated_fact["trimmed_exact_match_page_ids"]
                or validated_fact["whitespace_normalized_match_page_ids"]
            )
            if len(matched_page_ids) != 1:
                raise ValueError(
                    f"valid fact {source_id}:{fact_index} does not select one page"
                )
            page_id = str(matched_page_ids[0])
            indexed_documents = [
                document
                for document in title_to_documents[title]
                if str(document["page_id"]) == page_id
            ]
            if len(indexed_documents) != 1:
                raise ValueError(
                    f"page {page_id} is not uniquely indexed for title {title!r}"
                )
            document = indexed_documents[0]
            document_text = load_document(document["document_file"])
            raw_page = raw_pages_by_id[page_id]
            start, end, span_match_mode, alignment_mode = find_structural_evidence_span(
                document_text=document_text,
                raw_page=raw_page,
                context_sentences=context_by_title[title],
                sentence_index=sentence_index,
                comparison_sentence_text=validated_fact["comparison_sentence_text"],
                validation_status=status,
            )
            evidence_text = document_text[start:end]
            evidence_passage_ids.append(
                f"hotpot:{source_id}:fact_{fact_index}:page_{page_id}:sent_{sentence_index}"
            )
            evidence_texts.append(evidence_text)
            evidence_char_spans.append([start, end])
            evidence_page_ids.append(page_id)
            evidence_document_files.append(document["document_file"])
            evidence_titles.append(title)
            supporting_fact_sentence_indices.append(sentence_index)
            source_supporting_fact_texts.append(
                validated_fact["comparison_sentence_text"]
            )
            evidence_span_match_modes.append(span_match_mode)
            evidence_alignment_modes.append(alignment_mode)

        evidence_records.append(
            {
                "id": local_id,
                "source_id": source_id,
                "query": question,
                "answers": [answer],
                "evidence_passage_ids": evidence_passage_ids,
                "evidence_texts": evidence_texts,
                "evidence_char_spans": evidence_char_spans,
                "evidence_page_ids": evidence_page_ids,
                "evidence_document_files": evidence_document_files,
                "evidence_titles": evidence_titles,
                "supporting_fact_sentence_indices": supporting_fact_sentence_indices,
                "source_supporting_fact_texts": source_supporting_fact_texts,
                "evidence_span_match_modes": evidence_span_match_modes,
                "evidence_alignment_modes": evidence_alignment_modes,
                "invalid_supporting_facts": invalid_supporting_facts,
            }
        )
    return questions, answers, evidence_records, annotation_issues


def validate_records(
    *,
    dataset_root: Path,
    dev_records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not (
        len(dev_records) == len(questions) == len(answers) == len(evidence_records)
    ):
        raise ValueError("output record counts do not align")
    document_cache = {}
    evidence_count = 0
    span_mode_counts: Counter[str] = Counter()
    alignment_mode_counts: Counter[str] = Counter()
    query_issue_count = 0
    for local_id, (source, question, answer, evidence) in enumerate(
        zip(dev_records, questions, answers, evidence_records)
    ):
        if (
            question["id"] != local_id
            or answer["id"] != local_id
            or evidence["id"] != local_id
        ):
            raise ValueError(f"local ID mismatch at row {local_id}")
        if not (
            question["source_id"]
            == answer["source_id"]
            == evidence["source_id"]
            == source["_id"]
        ):
            raise ValueError(f"source ID mismatch at row {local_id}")
        if (
            question["query"] != source["question"]
            or evidence["query"] != source["question"]
        ):
            raise ValueError(f"query mismatch at row {local_id}")
        aligned_fields = [
            "evidence_passage_ids",
            "evidence_texts",
            "evidence_char_spans",
            "evidence_page_ids",
            "evidence_document_files",
            "evidence_titles",
            "supporting_fact_sentence_indices",
            "source_supporting_fact_texts",
            "evidence_span_match_modes",
            "evidence_alignment_modes",
        ]
        lengths = {len(evidence[field]) for field in aligned_fields}
        if len(lengths) != 1:
            raise ValueError(f"evidence field length mismatch at row {local_id}")
        evidence_count += len(evidence["evidence_texts"])
        span_mode_counts.update(evidence["evidence_span_match_modes"])
        alignment_mode_counts.update(evidence["evidence_alignment_modes"])
        query_issue_count += int(bool(evidence["invalid_supporting_facts"]))
        for text, span, document_file in zip(
            evidence["evidence_texts"],
            evidence["evidence_char_spans"],
            evidence["evidence_document_files"],
        ):
            if document_file not in document_cache:
                document_cache[document_file] = (
                    dataset_root / document_file
                ).read_text(encoding="utf-8")
            start, end = span
            if document_cache[document_file][start:end] != text:
                raise ValueError(f"evidence span mismatch at row {local_id}")
    return {
        "query_count": len(questions),
        "answer_count": len(answers),
        "evidence_record_count": len(evidence_records),
        "valid_evidence_fact_count": evidence_count,
        "query_with_annotation_issue_count": query_issue_count,
        "span_match_mode_counts": dict(span_mode_counts),
        "alignment_mode_counts": dict(alignment_mode_counts),
    }


def build_rag_inputs(root: Path) -> dict[str, Any]:
    """Build the query, answer, and evidence files under ``root``."""

    output_paths = [
        root / "questions/query.jsonl",
        root / "answers/answer.jsonl",
        root / "dataset_info/evidence_labels.jsonl",
        root / "dataset_info/rag_input_validation_manifest.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    with (root / "hotpot_dev_distractor_v1.json").open("r", encoding="utf-8") as handle:
        dev_records = json.load(handle)
    validation_records = load_jsonl(
        root / "dataset_info/supporting_fact_validation.jsonl"
    )
    with (root / "dataset_info/title_to_documents.json").open(
        "r", encoding="utf-8"
    ) as handle:
        title_to_documents = json.load(handle)
    selected_page_ids = {
        page_id
        for record in validation_records
        for fact in record["supporting_facts"]
        for page_id in (
            fact["trimmed_exact_match_page_ids"]
            or fact["whitespace_normalized_match_page_ids"]
        )
    }
    raw_pages_by_id = load_selected_raw_pages(
        root / "dataset_info/documents_raw.jsonl", selected_page_ids
    )
    questions, answers, evidence_records, annotation_issues = build_records(
        dataset_root=root,
        dev_records=dev_records,
        validation_records=validation_records,
        title_to_documents=title_to_documents,
        raw_pages_by_id=raw_pages_by_id,
    )
    counts = validate_records(
        dataset_root=root,
        dev_records=dev_records,
        questions=questions,
        answers=answers,
        evidence_records=evidence_records,
    )
    expected_counts = {
        "query_count": 7405,
        "answer_count": 7405,
        "evidence_record_count": 7405,
        "valid_evidence_fact_count": 18003,
        "query_with_annotation_issue_count": 2,
        "span_match_mode_counts": {
            "trimmed_exact": 17519,
            "whitespace_normalized": 484,
        },
        "alignment_mode_counts": {
            "source_paragraph_exact": 17025,
            "source_paragraph_whitespace_normalized": 976,
            "source_sentence_index_max_agreement": 2,
        },
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected output counts: {counts}")
    if len(annotation_issues) != 2:
        raise ValueError(
            f"expected two source annotation issues, got {len(annotation_issues)}"
        )

    write_jsonl_atomic(output_paths[0], questions)
    write_jsonl_atomic(output_paths[1], answers)
    write_jsonl_atomic(output_paths[2], evidence_records)
    output_hashes = {
        str(path.relative_to(root)): file_sha256(path) for path in output_paths[:3]
    }
    manifest = {
        "dataset": "custom_hotpotqa_distractor_dev_full_wikipedia_documents",
        "source_split": "official HotpotQA distractor development",
        "custom_protocol": True,
        "id_policy": (
            "contiguous local IDs 0..7404 in unchanged official dev order; "
            "official _id preserved as source_id"
        ),
        "answer_policy": (
            "preserve the single official answer string verbatim and expose it "
            "as one answers entry"
        ),
        "evidence_policy": {
            "source": "official supporting_facts title and sentence index",
            "character_span": "half-open [start, end) in the stored document file",
            "evidence_text": "exact stored document substring at that span",
            "whitespace_alignment": (
                "normalized matches are reverse-mapped to the original document "
                "character range"
            ),
            "duplicate_sentence_disambiguation": (
                "prefer complete official-context-to-source-paragraph alignment; "
                "if a sibling annotation is malformed, require a unique same-length "
                "source paragraph with the labeled sentence matching at the official "
                "index and maximal positional agreement"
            ),
            "alternative_page_policy": (
                "the validated supporting sentence uniquely identifies one page for "
                "every valid fact"
            ),
            "fuzzy_matching": False,
            "invalid_source_annotations_are_excluded_from_evidence_texts": True,
        },
        "counts": counts,
        "source_annotation_issues": annotation_issues,
        "outputs": output_hashes,
    }
    write_json_atomic(output_paths[3], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_rag_inputs(args.dataset_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
