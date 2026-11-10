"""Expose the existing custom full COVID-QA set to the common pipeline."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dataset.prepare_covid_qa_full_test import URL
from dataset.preparation import (
    download_file,
    file_sha256,
    staging_directory,
    write_json,
    write_jsonl,
)

SOURCE_SHA256 = "291abf17f4bc2bd343838fd8ef5debb6278bbbb61b262db1f1bd58048fff76b9"


def prepare(source: Path, output: Path) -> dict:
    staging = staging_directory(output)
    work = staging / "dataset"
    (work / "raw").mkdir(parents=True)
    shutil.copyfile(source, work / "raw/COVID-QA.json")
    native = Path(__file__).resolve().parents[1] / "prepare_covid_qa_full_test.py"
    subprocess.run([sys.executable, str(native), "--output", str(work)], check=True)
    original = json.loads(source.read_text())
    docs = work / "documents"
    docs.mkdir()
    for document in original["data"]:
        for paragraph in document["paragraphs"]:
            target = docs / f"{paragraph['document_id']}.txt"
            if target.exists():
                if target.read_text() != paragraph["context"]:
                    raise ValueError(f"conflicting source document: {target.name}")
            else:
                target.write_text(paragraph["context"], encoding="utf-8")
    with (work / "test.jsonl").open() as handle:
        records = [json.loads(line) for line in handle]
    questions = [{"id": row["id"], "query": row["question"]} for row in records]
    answers = [
        {"id": row["id"], "answers": [answer["text"] for answer in row["answers"]]}
        for row in records
    ]
    write_jsonl(work / "questions/query.jsonl", questions)
    write_jsonl(work / "answers/answer.jsonl", answers)
    manifest = {
        "custom_protocol": True,
        "scope": "Existing full-set COVID-QA custom test and shared original-context corpus",
        "questions": len(questions),
        "documents": len(list(docs.iterdir())),
        "source_sha256": file_sha256(source),
        "questions_and_answers": "Original order, IDs, question strings, and every answer text",
        "native_source_preserved": True,
    }
    if manifest["documents"] != 147:
        raise ValueError(f"expected 147 document contexts: {manifest['documents']}")
    write_json(work / "dataset_info/manifest.json", manifest)
    if output.exists():
        raise FileExistsError(output)
    work.rename(output)
    staging.rmdir()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.source_file is not None:
        source = args.source_file.resolve()
    else:
        source_dir = args.source_dir or output.with_name(output.name + "-sources")
        source = download_file(URL, source_dir / "COVID-QA.json", SOURCE_SHA256)
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError(f"source does not match the existing COVID-QA input: {source}")
    manifest = prepare(source, output)
    print(
        f"Prepared {manifest['questions']} questions and {manifest['documents']} documents"
    )


if __name__ == "__main__":
    main()
