"""Preserve official COVID-QA and expose every question as a custom test set."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

URL = (
    "https://raw.githubusercontent.com/deepset-ai/COVID-QA/"
    "master/data/question-answering/COVID-QA.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    (root / "raw").mkdir(parents=True, exist_ok=True)
    source = root / "raw/COVID-QA.json"
    if not source.exists():
        raw = urllib.request.urlopen(URL, timeout=120).read()
        json.loads(raw)
        source.write_bytes(raw)
    raw = source.read_bytes()
    data = json.loads(raw)
    records, issues = [], []
    for di, document in enumerate(data["data"]):
        for pi, paragraph in enumerate(document["paragraphs"]):
            for question in paragraph["qas"]:
                assert not (question.keys() & {"context", "document_id"})
                record = {
                    **question,
                    "context": paragraph["context"],
                    "document_id": paragraph["document_id"],
                }
                records.append(record)
                for ai, answer in enumerate(question["answers"]):
                    start = answer["answer_start"]
                    text = answer["text"]
                    actual = paragraph["context"][start : start + len(text)]
                    if actual != text:
                        issues.append(
                            {
                                "document_index": di,
                                "paragraph_index": pi,
                                "question_id": question["id"],
                                "answer_index": ai,
                                "answer_start": start,
                                "stored_text": text,
                                "context_slice": actual,
                            }
                        )
    assert len(records) == 2019
    assert len({r["id"] for r in records}) == len(records)
    (root / "test.json").write_bytes(raw)
    (root / "test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    saved = [
        json.loads(line)
        for line in (root / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert saved == records
    assert (root / "test.json").read_bytes() == raw
    (root / "answer_offset_audit.json").write_text(
        json.dumps(issues, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": "deepset COVID-QA",
        "split": "custom full-set test; not an official held-out test split",
        "source_url": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "document_entries": len(data["data"]),
        "unique_document_ids": len({r["document_id"] for r in records}),
        "questions": len(records),
        "answers": sum(len(r["answers"]) for r in records),
        "answer_offset_mismatches": len(issues),
        "all_questions_retained": True,
        "native_test_json_byte_identical": True,
        "jsonl": "Every original question field plus verbatim context and document_id; source order.",
        "preprocessing": "None: no filtering, splitting, normalization, repair, or answer transformation.",
        "evaluation_setting": "User requested all questions as test; paired provided context, no retrieval.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
