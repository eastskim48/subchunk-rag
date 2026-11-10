"""Prepare the existing custom 4,293-question NewsQA retrieval dataset."""

import argparse
from pathlib import Path
import subprocess

from dataset.newsqa import consensus, documents
from dataset.preparation import (
    download_file,
    file_sha256,
    staging_directory,
    write_json,
    write_jsonl,
)

OFFICIAL_REVISION = "d5bb9e9640e2ed7a31e209393376549d737d276b"
OFFICIAL_FILES = (
    "stories_requiring_extra_newline.csv",
    "stories_requiring_two_extra_newlines.csv",
    "stories_to_decode_specially.csv",
    "test_story_ids.csv",
)
ANNOTATIONS_URL = "https://download.microsoft.com/download/1/d/8/1d830cee-f8d1-4807-9224-de35a8f08dc4/newsqa-data-v1.zip"
ANNOTATIONS_SHA256 = "5b542995d1d46efd9d90b16a5aee3b04f3fe17c9ee2a2d5094e9f978175357bd"
STORIES_URL = "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/11343c3752184397d56efc19a8a7cceb68089318/data/cnn_stories.tgz"
STORIES_SHA256 = "e8fbc0027e54e0a916abd9c969eb35f708ed1467d7ef4e3b17a56739d65cb200"


def source_archives(args, source_dir: Path) -> tuple[Path, Path]:
    for path in (args.annotations_zip, args.stories_archive):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    annotations = args.annotations_zip or download_file(
        args.annotations_url,
        source_dir / "newsqa-data-v1.zip",
        ANNOTATIONS_SHA256,
        reject_html=True,
    )
    stories = args.stories_archive or download_file(
        args.stories_url,
        source_dir / "cnn_stories.tgz",
        STORIES_SHA256,
        reject_html=True,
    )
    return annotations.resolve(), stories.resolve()


def official_files(source_dir: Path, repository: Path | None) -> Path:
    if repository is not None:
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != OFFICIAL_REVISION:
            raise ValueError(
                f"official NewsQA repository must be at {OFFICIAL_REVISION}"
            )
        folder = repository / "maluuba/newsqa"
        for name in OFFICIAL_FILES:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        return folder
    folder = source_dir / OFFICIAL_REVISION
    for name in OFFICIAL_FILES:
        download_file(
            f"https://raw.githubusercontent.com/Maluuba/newsqa/{OFFICIAL_REVISION}/maluuba/newsqa/{name}",
            folder / name,
        )
    return folder


def prepare(
    annotations: Path,
    stories: Path,
    official: Path,
    output: Path,
) -> dict:
    staging = staging_directory(output)
    work = staging / "dataset"
    texts = documents.load_story_texts(
        stories,
        documents.load_story_ids(annotations),
        extra_newline_ids=documents.load_lines(
            official / "stories_requiring_extra_newline.csv"
        ),
        two_extra_newline_ids=documents.load_lines(
            official / "stories_requiring_two_extra_newlines.csv"
        ),
        special_decode_ids=documents.load_lines(
            official / "stories_to_decode_specially.csv"
        ),
    )
    documents.validate_known_official_examples(texts)
    documents.write_documents(
        work, texts, annotations_zip=annotations, stories_archive=stories
    )
    rows = consensus.load_raw_answerable_test_rows(
        annotations, consensus.load_test_story_ids(official / "test_story_ids.csv")
    )
    raw_questions = [
        {"id": index, "query": row["question"]} for index, row in enumerate(rows)
    ]
    questions, answers = consensus.build_consensus_records(
        rows, raw_questions, work / "documents"
    )
    for suffix in ("", "_full"):
        write_jsonl(work / f"questions/query{suffix}.jsonl", questions)
        write_jsonl(work / f"answers/answer{suffix}.jsonl", answers)
    manifest = {
        "custom_protocol": True,
        "scope": "Existing NewsQA single-consensus test population; all annotated articles",
        "questions": len(questions),
        "documents": len(texts),
        "raw_answerable_test_rows": len(rows),
        "official_revision": OFFICIAL_REVISION,
        "source_sha256": {
            str(path): file_sha256(path)
            for path in [annotations, stories, *(official / n for n in OFFICIAL_FILES)]
        },
        "question_ids": "Original raw-test-row IDs and original order",
        "answers": "Existing official_consensus_span implementation, unchanged",
        "sampling": "None; all 4293 existing consensus records are active",
    }
    write_json(work / "dataset_info/qa_manifest.json", manifest)
    if output.exists():
        raise FileExistsError(output)
    work.rename(output)
    staging.rmdir()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", type=Path)
    parser.add_argument("--stories-archive", type=Path)
    parser.add_argument("--annotations-url", default=ANNOTATIONS_URL)
    parser.add_argument("--stories-url", default=STORIES_URL)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = args.source_dir or output.with_name(output.name + "-sources")
    annotations, stories = source_archives(args, source)
    official = official_files(source, args.official_repo_dir)
    manifest = prepare(annotations, stories, official, output)
    print(
        f"Prepared {manifest['questions']} questions and {manifest['documents']} documents"
    )


if __name__ == "__main__":
    main()
