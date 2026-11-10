#!/usr/bin/env python
"""Build the custom DAPR-NQ/NQ-open development-intersection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

NQ_OPEN_DEV_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "natural-questions/a7d6452c0905c7772e9fbbb9a20b5fcab07c668f/"
    "nq_open/NQ-open.dev.jsonl"
)
DAPR_DATASET = "UKPLab/dapr"
DAPR_CONFIG = "NaturalQuestions"
DAPR_SNAPSHOT_REVISION = "67ae3daa13596700976d20605630f5f9db3bd732"

EXPECTED_DAPR_DOCUMENTS = 108_626
EXPECTED_DAPR_PASSAGES = 2_682_017
EXPECTED_DAPR_QUERIES = 3_610
EXPECTED_DAPR_QRELS = 4_379
EXPECTED_NQ_OPEN_QUERIES = 3_610
EXPECTED_INTERSECTION_QUERIES = 2_390
EXPECTED_INTERSECTION_QRELS = 2_971

SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def load_parquet_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def join_queries_and_answers(
    dapr_queries: list[dict[str, Any]],
    nq_open_records: list[dict[str, Any]],
) -> dict[str, Any]:
    nq_by_question: dict[str, dict[str, Any]] = {}
    for record in nq_open_records:
        question = record.get("question")
        answers = record.get("answer")
        if not isinstance(question, str) or not question:
            raise ValueError(f"invalid NQ-open question: {question!r}")
        if question in nq_by_question:
            raise ValueError(f"duplicate NQ-open question: {question!r}")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer for answer in answers)
        ):
            raise ValueError(f"invalid NQ-open answers for question {question!r}")
        nq_by_question[question] = record

    seen_dapr_ids: set[str] = set()
    seen_dapr_questions: set[str] = set()
    questions = []
    answers = []
    source_id_to_local_id: dict[str, int] = {}
    source_id_to_answers: dict[str, list[str]] = {}
    for dapr_record in dapr_queries:
        source_id = str(dapr_record["_id"])
        question = dapr_record["text"]
        if not isinstance(question, str) or not question:
            raise ValueError(f"invalid DAPR question for {source_id!r}")
        if source_id in seen_dapr_ids:
            raise ValueError(f"duplicate DAPR query ID: {source_id!r}")
        if question in seen_dapr_questions:
            raise ValueError(f"duplicate DAPR question: {question!r}")
        seen_dapr_ids.add(source_id)
        seen_dapr_questions.add(question)

        nq_record = nq_by_question.get(question)
        if nq_record is None:
            continue
        local_id = len(questions)
        answer_aliases = list(nq_record["answer"])
        questions.append({"id": local_id, "query": question})
        answers.append({"id": local_id, "answers": answer_aliases})
        source_id_to_local_id[source_id] = local_id
        source_id_to_answers[source_id] = answer_aliases

    return {
        "questions": questions,
        "answers": answers,
        "source_id_to_local_id": source_id_to_local_id,
        "source_id_to_answers": source_id_to_answers,
        "nq_question_count": len(nq_by_question),
        "dapr_query_count": len(seen_dapr_ids),
    }


def filter_intersection_qrels(
    dapr_qrels: list[dict[str, Any]],
    selected_source_ids: set[str],
) -> dict[str, Any]:
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for qrel in dapr_qrels:
        query_id = str(qrel["query_id"])
        passage_id = str(qrel["corpus_id"])
        pair = (query_id, passage_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate DAPR qrel: {pair!r}")
        seen_pairs.add(pair)
        if query_id not in selected_source_ids:
            continue
        score = qrel["score"]
        if isinstance(score, bool) or not isinstance(score, int) or score <= 0:
            raise ValueError(f"invalid DAPR qrel score for {pair!r}: {score!r}")
        qrels_by_query[query_id].append(
            {
                "query_id": query_id,
                "passage_id": passage_id,
                "score": score,
            }
        )

    missing = selected_source_ids - set(qrels_by_query)
    if missing:
        raise ValueError(
            "selected DAPR queries without qrels: " f"{sorted(missing)[:5]}"
        )
    return {
        "qrels_by_query": dict(qrels_by_query),
        "selected_qrel_count": sum(map(len, qrels_by_query.values())),
        "dapr_qrel_count": len(seen_pairs),
        "wanted_passage_ids": {
            qrel["passage_id"] for qrels in qrels_by_query.values() for qrel in qrels
        },
    }


def serialize_document(
    title: str,
    passage_ids: list[str],
    passages: list[str],
    wanted_passage_ids: set[str] | None = None,
) -> tuple[str, dict[str, tuple[int, int]]]:
    if not isinstance(title, str):
        raise ValueError(f"document title is not a string: {title!r}")
    if len(passage_ids) != len(passages):
        raise ValueError(
            "passage_ids/passages length mismatch: "
            f"{len(passage_ids)} != {len(passages)}"
        )
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("duplicate passage ID in one document")
    if not all(isinstance(passage, str) for passage in passages):
        raise ValueError("document contains a non-string passage")

    wanted = wanted_passage_ids or set()
    parts = [title]
    spans: dict[str, tuple[int, int]] = {}
    cursor = len(title)
    for passage_index, (passage_id, passage) in enumerate(zip(passage_ids, passages)):
        separator = "\n" if passage_index == 0 else "\n\n"
        parts.append(separator)
        cursor += len(separator)
        start = cursor
        parts.append(passage)
        cursor += len(passage)
        if passage_id in wanted:
            spans[passage_id] = (start, cursor)
    return "".join(parts), spans


def build_evidence_records(
    questions: list[dict[str, Any]],
    source_id_to_local_id: dict[str, int],
    source_id_to_answers: dict[str, list[str]],
    qrels_by_query: dict[str, list[dict[str, Any]]],
    passage_locations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_id_by_local_id = {
        local_id: source_id for source_id, local_id in source_id_to_local_id.items()
    }
    evidence_records = []
    for question in questions:
        local_id = int(question["id"])
        source_id = source_id_by_local_id[local_id]
        qrels = qrels_by_query[source_id]
        locations = [passage_locations[qrel["passage_id"]] for qrel in qrels]
        gold_doc_ids = {location["doc_id"] for location in locations}
        if len(gold_doc_ids) != 1:
            raise ValueError(
                f"query {source_id!r} has qrels in multiple parent documents: "
                f"{sorted(gold_doc_ids)}"
            )
        first = locations[0]
        evidence_records.append(
            {
                "id": local_id,
                "source_id": source_id,
                "query": question["query"],
                "answers": source_id_to_answers[source_id],
                "gold_doc_id": first["doc_id"],
                "gold_doc_title": first["title"],
                "document_file": first["document_file"],
                "evidence_passage_ids": [qrel["passage_id"] for qrel in qrels],
                "evidence_qrel_scores": [qrel["score"] for qrel in qrels],
                "evidence_texts": [location["text"] for location in locations],
                "evidence_char_spans": [
                    list(location["char_span"]) for location in locations
                ],
            }
        )
    return evidence_records


def iter_document_records(doc_shards: list[Path]):
    for shard in doc_shards:
        parquet_file = pq.ParquetFile(shard)
        for batch in parquet_file.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def materialize_documents(
    *,
    documents_dir: Path,
    doc_shards: list[Path],
    wanted_passage_ids: set[str],
) -> dict[str, Any]:
    documents_dir.mkdir(parents=True)
    seen_doc_ids: set[str] = set()
    passage_locations: dict[str, dict[str, Any]] = {}
    passage_count = 0
    candidate_count = 0

    for document_count, document in enumerate(
        iter_document_records(doc_shards), start=1
    ):
        doc_id = str(document["doc_id"])
        if doc_id in seen_doc_ids:
            raise ValueError(f"duplicate DAPR document ID: {doc_id!r}")
        if not SAFE_DOCUMENT_ID.fullmatch(doc_id):
            raise ValueError(f"unsafe DAPR document ID: {doc_id!r}")
        seen_doc_ids.add(doc_id)

        title = document["title"]
        passage_ids = [str(value) for value in document["passage_ids"]]
        passages = document["passages"]
        candidates = document["is_candidate"]
        if len(candidates) != len(passage_ids):
            raise ValueError(f"is_candidate length mismatch for document {doc_id!r}")
        if not all(isinstance(value, bool) for value in candidates):
            raise ValueError(f"invalid is_candidate value in document {doc_id!r}")
        candidate_count += sum(candidates)
        passage_count += len(passage_ids)

        text, spans = serialize_document(
            title=title,
            passage_ids=passage_ids,
            passages=passages,
            wanted_passage_ids=wanted_passage_ids,
        )
        document_file = f"doc_{doc_id}.txt"
        (documents_dir / document_file).write_text(text, encoding="utf-8")

        passage_text_by_id = dict(zip(passage_ids, passages))
        for passage_id, char_span in spans.items():
            if passage_id in passage_locations:
                raise ValueError(f"duplicate qrel passage ID in corpus: {passage_id!r}")
            start, end = char_span
            passage_text = passage_text_by_id[passage_id]
            if text[start:end] != passage_text:
                raise ValueError(
                    f"serialized evidence span mismatch for passage {passage_id!r}"
                )
            passage_locations[passage_id] = {
                "doc_id": doc_id,
                "document_file": document_file,
                "title": title,
                "text": passage_text,
                "char_span": char_span,
            }

        if document_count % 5_000 == 0:
            print(
                f"materialized {document_count:,} documents, "
                f"{passage_count:,} passages",
                flush=True,
            )

    missing_passages = wanted_passage_ids - set(passage_locations)
    if missing_passages:
        raise ValueError(
            "qrel passages absent from DAPR documents: "
            f"{sorted(missing_passages)[:5]}"
        )
    return {
        "document_count": len(seen_doc_ids),
        "passage_count": passage_count,
        "candidate_count": candidate_count,
        "passage_locations": passage_locations,
    }


def validate_output(
    *,
    staging_dir: Path,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
) -> None:
    if not (len(questions) == len(answers) == len(evidence_records)):
        raise ValueError("query, answer, and evidence-label counts differ")

    document_cache: dict[str, str] = {}
    for question, answer, evidence in zip(questions, answers, evidence_records):
        local_id = question["id"]
        if answer["id"] != local_id or evidence["id"] != local_id:
            raise ValueError(f"unaligned local ID: {local_id}")
        if question["query"] != evidence["query"]:
            raise ValueError(f"unaligned query for local ID {local_id}")
        if answer["answers"] != evidence["answers"]:
            raise ValueError(f"unaligned answers for local ID {local_id}")

        document_file = evidence["document_file"]
        document_text = document_cache.get(document_file)
        if document_text is None:
            document_text = (staging_dir / "documents" / document_file).read_text(
                encoding="utf-8"
            )
            document_cache[document_file] = document_text
        for evidence_text, (start, end) in zip(
            evidence["evidence_texts"], evidence["evidence_char_spans"]
        ):
            if document_text[start:end] != evidence_text:
                raise ValueError(
                    f"output evidence span mismatch for local ID {local_id}"
                )


def build_dataset(
    *,
    dapr_root: Path,
    nq_open_dev_file: Path,
    output_dir: Path,
    skip_official_count_check: bool = False,
) -> dict[str, Any]:
    query_path = dapr_root / "queries" / "test.parquet"
    qrel_path = dapr_root / "qrels" / "test.parquet"
    doc_shards = sorted((dapr_root / "docs").glob("test-*.parquet"))
    if not doc_shards:
        raise FileNotFoundError(f"no DAPR document shards under {dapr_root / 'docs'}")

    dapr_queries = load_parquet_records(query_path)
    dapr_qrels = load_parquet_records(qrel_path)
    nq_open_records = load_jsonl(nq_open_dev_file)
    joined = join_queries_and_answers(dapr_queries, nq_open_records)
    filtered_qrels = filter_intersection_qrels(
        dapr_qrels,
        set(joined["source_id_to_local_id"]),
    )

    observed_small_counts = {
        "dapr_queries": joined["dapr_query_count"],
        "dapr_qrels": filtered_qrels["dapr_qrel_count"],
        "nq_open_queries": joined["nq_question_count"],
        "intersection_queries": len(joined["questions"]),
        "intersection_qrels": filtered_qrels["selected_qrel_count"],
    }
    expected_small_counts = {
        "dapr_queries": EXPECTED_DAPR_QUERIES,
        "dapr_qrels": EXPECTED_DAPR_QRELS,
        "nq_open_queries": EXPECTED_NQ_OPEN_QUERIES,
        "intersection_queries": EXPECTED_INTERSECTION_QUERIES,
        "intersection_qrels": EXPECTED_INTERSECTION_QRELS,
    }
    if not skip_official_count_check and observed_small_counts != expected_small_counts:
        raise ValueError(
            "unexpected official query/qrel counts: "
            f"observed={observed_small_counts}, expected={expected_small_counts}"
        )

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        document_summary = materialize_documents(
            documents_dir=staging_dir / "documents",
            doc_shards=doc_shards,
            wanted_passage_ids=filtered_qrels["wanted_passage_ids"],
        )
        observed_document_counts = {
            "documents": document_summary["document_count"],
            "passages": document_summary["passage_count"],
            "candidate_passages": document_summary["candidate_count"],
        }
        expected_document_counts = {
            "documents": EXPECTED_DAPR_DOCUMENTS,
            "passages": EXPECTED_DAPR_PASSAGES,
            "candidate_passages": EXPECTED_DAPR_PASSAGES,
        }
        if (
            not skip_official_count_check
            and observed_document_counts != expected_document_counts
        ):
            raise ValueError(
                "unexpected official document counts: "
                f"observed={observed_document_counts}, "
                f"expected={expected_document_counts}"
            )

        evidence_records = build_evidence_records(
            questions=joined["questions"],
            source_id_to_local_id=joined["source_id_to_local_id"],
            source_id_to_answers=joined["source_id_to_answers"],
            qrels_by_query=filtered_qrels["qrels_by_query"],
            passage_locations=document_summary["passage_locations"],
        )
        validate_output(
            staging_dir=staging_dir,
            questions=joined["questions"],
            answers=joined["answers"],
            evidence_records=evidence_records,
        )

        write_jsonl(
            staging_dir / "questions" / "query.jsonl",
            joined["questions"],
        )
        write_jsonl(
            staging_dir / "answers" / "answer.jsonl",
            joined["answers"],
        )
        write_jsonl(
            staging_dir / "dataset_info" / "evidence_labels.jsonl",
            evidence_records,
        )

        input_files = [*doc_shards, query_path, qrel_path, nq_open_dev_file]
        manifest = {
            "format": "dapr_nq_nq_open_intersection_rag",
            "format_version": 1,
            "source": {
                "dapr_dataset": DAPR_DATASET,
                "dapr_config": DAPR_CONFIG,
                "dapr_snapshot_revision": DAPR_SNAPSHOT_REVISION,
                "dapr_split": "test",
                "nq_open_dev_url": NQ_OPEN_DEV_URL,
                "input_sha256": {
                    (
                        path.name
                        if path == nq_open_dev_file
                        else str(path.relative_to(dapr_root))
                    ): file_sha256(path)
                    for path in input_files
                },
            },
            "selection": (
                "verbatim exact-question intersection of DAPR-NQ test and "
                "official NQ-open dev, ordered by DAPR-NQ test"
            ),
            "document_serialization": (
                'title + "\\n" + passages joined with "\\n\\n"; '
                "all DAPR parent documents and passages retained; no chunking"
            ),
            "answer_semantics": (
                "complete official NQ-open answer alias list, copied verbatim"
            ),
            "evidence_semantics": (
                "positive DAPR test qrel passages with half-open Python "
                "character spans in serialized parent documents"
            ),
            "counts": {
                **observed_small_counts,
                **observed_document_counts,
                "intersection_gold_parent_documents": len(
                    {evidence["gold_doc_id"] for evidence in evidence_records}
                ),
                "answer_alias_count_distribution": dict(
                    sorted(
                        Counter(
                            len(answer["answers"]) for answer in joined["answers"]
                        ).items()
                    )
                ),
            },
            "paths": {
                "questions": "questions/query.jsonl",
                "answers": "answers/answer.jsonl",
                "documents": "documents/doc_{dapr_doc_id}.txt",
                "evidence_labels": "dataset_info/evidence_labels.jsonl",
            },
        }
        manifest_path = staging_dir / "dataset_info" / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapr-root", type=Path, required=True)
    parser.add_argument("--nq-open-dev-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="Allow non-official-size fixtures for converter development only.",
    )
    args = parser.parse_args()

    manifest = build_dataset(
        dapr_root=args.dapr_root,
        nq_open_dev_file=args.nq_open_dev_file,
        output_dir=args.output_dir,
        skip_official_count_check=args.skip_official_count_check,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
