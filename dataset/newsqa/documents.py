#!/usr/bin/env python
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

EXPECTED_DOCUMENTS = 12_744
CSV_MEMBER = "newsqa-data-v1/newsqa-data-v1.csv"
COPYRIGHT_LINE_PATTERN = re.compile(
    r"^(Copyright|Entire contents of this article copyright, )"
)


def load_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_story_ids(annotations_zip: Path) -> set[str]:
    with zipfile.ZipFile(annotations_zip) as archive:
        with archive.open(CSV_MEMBER) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
            story_ids = {row["story_id"] for row in csv.DictReader(text_handle)}
    if len(story_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"expected {EXPECTED_DOCUMENTS} NewsQA story IDs, got {len(story_ids)}"
        )
    return story_ids


def strip_empty_strings(lines: list[str]) -> list[str]:
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def decode_story(
    raw_story: bytes,
    *,
    decode_specially: bool,
    extra_newlines: int,
) -> str:
    raw_lines = raw_story.splitlines()
    encoding = "latin-1" if decode_specially else "utf-8"
    lines = [raw_line.strip().decode(encoding) for raw_line in raw_lines]
    highlights_start = lines.index("@highlight")
    story_lines = strip_empty_strings(lines[:highlights_start])
    while len(story_lines) > 1 and COPYRIGHT_LINE_PATTERN.search(story_lines[-1]):
        story_lines = strip_empty_strings(story_lines[:-2])

    story_text = ("\n" * (extra_newlines + 1)).join(story_lines)
    story_text = story_text.replace("\xe2\x80\xa2", "\xe2\u20ac\xa2")
    story_text = story_text.replace("\xe2\x82\xac", "\xe2\u201a\xac")
    story_text = story_text.replace("\r", "\n")
    if decode_specially:
        story_text = story_text.replace("\xe9", "\xc3\xa9")
    return story_text


def load_story_texts(
    stories_archive: Path,
    story_ids: set[str],
    *,
    extra_newline_ids: set[str],
    two_extra_newline_ids: set[str],
    special_decode_ids: set[str],
) -> dict[str, str]:
    remaining = set(story_ids)
    story_texts: dict[str, str] = {}
    with tarfile.open(stories_archive, mode="r:gz") as archive:
        for member in archive:
            if member.name not in remaining:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read story archive member: {member.name}")
            extra_newlines = (
                2
                if member.name in two_extra_newline_ids
                else 1 if member.name in extra_newline_ids else 0
            )
            story_texts[member.name] = decode_story(
                handle.read(),
                decode_specially=member.name in special_decode_ids,
                extra_newlines=extra_newlines,
            )
            remaining.remove(member.name)
            if not remaining:
                break
    if remaining:
        raise ValueError(
            f"story archive is missing {len(remaining)} NewsQA documents; "
            f"examples={sorted(remaining)[:5]}"
        )
    return story_texts


def validate_known_official_examples(story_texts: dict[str, str]) -> None:
    expected_prefixes = {
        "./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story": (
            "NEW DELHI, India (CNN) -- A high court in nort"
        ),
        "./cnn/stories/d312173b8c95cc6c206a32cc0acd8a92c1e272d5.story": (
            "CNN affiliates report on where job seekers are"
        ),
        "./cnn/stories/d41dc7fc05273a37f0aceaf4f3e35a187f12653e.story": (
            "(CNN) -- A former government contract"
        ),
    }
    for story_id, expected_prefix in expected_prefixes.items():
        observed = story_texts[story_id][: len(expected_prefix)]
        if observed != expected_prefix:
            raise ValueError(
                f"official NewsQA story reconstruction mismatch for {story_id}: "
                f"{observed!r} != {expected_prefix!r}"
            )

    expected_spans = {
        (
            "./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story",
            294,
            297,
        ): "19 ",
        (
            "./cnn/stories/d312173b8c95cc6c206a32cc0acd8a92c1e272d5.story",
            301,
            324,
        ): "the federal government ",
        (
            "./cnn/stories/d41dc7fc05273a37f0aceaf4f3e35a187f12653e.story",
            303,
            320,
        ): "Roy Lynn Oakley, ",
    }
    for (story_id, start, end), expected_text in expected_spans.items():
        observed = story_texts[story_id][start:end]
        if observed != expected_text:
            raise ValueError(
                f"official NewsQA character-offset mismatch for {story_id} "
                f"[{start}:{end}]: {observed!r} != {expected_text!r}"
            )


def write_documents(
    output_dir: Path,
    story_texts: dict[str, str],
    *,
    annotations_zip: Path,
    stories_archive: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        documents_dir = staging_dir / "documents"
        documents_dir.mkdir()
        for story_id, text in sorted(story_texts.items()):
            document_file = f"doc_{Path(story_id).stem}.txt"
            (documents_dir / document_file).write_text(text, encoding="utf-8")

        dataset_info_dir = staging_dir / "dataset_info"
        dataset_info_dir.mkdir()
        manifest = {
            "format": "newsqa_full_article_corpus",
            "format_version": 1,
            "num_documents": len(story_texts),
            "document_content": (
                "CNN article body before the first @highlight marker, reconstructed "
                "with the official Maluuba NewsQA correction rules"
            ),
            "source_annotations": str(annotations_zip),
            "source_stories": str(stories_archive),
            "paths": {"documents": "documents/doc_{CNN_story_SHA1}.txt"},
        }
        (dataset_info_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir)
        raise
