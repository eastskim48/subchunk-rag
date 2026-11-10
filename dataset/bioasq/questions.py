"""Prepare the user-requested custom BioASQ subset excluding summary questions.

Keep original annotations and IDs. This does not retrieve, generate answers,
flatten list references, or parse generated list answers.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def prepare(source: Path, output: Path):
    raw = source.read_bytes()
    original = json.loads(raw)
    questions = original["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source question IDs")
    counts = Counter(q["type"] for q in questions)
    unknown = set(counts) - {"yesno", "factoid", "list", "summary"}
    if unknown:
        raise ValueError(f"Unknown question types: {unknown}")
    selected = [q for q in questions if q["type"] != "summary"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "questions").mkdir()
    (output / "answers").mkdir()
    (output / "original.json").write_bytes(raw)
    (output / "gold.json").write_text(
        json.dumps({**original, "questions": selected}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output / "questions/query.jsonl").write_text(
        "".join(
            json.dumps({"id": q["id"], "query": q["body"]}, ensure_ascii=False) + "\n"
            for q in selected
        ),
        encoding="utf-8",
    )
    # Preserve nested exact_answer items/synonyms for the later approved scorer.
    (output / "answers/answer.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in selected),
        encoding="utf-8",
    )
    (output / "audit.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "included": q["type"] != "summary",
                },
                ensure_ascii=False,
            )
            + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    manifest = {
        "scope": "custom BioASQ subset, excluding only type=summary by user instruction",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_questions": len(questions),
        "selected_questions": len(selected),
        "source_types": dict(counts),
        "selected_types": dict(Counter(q["type"] for q in selected)),
        "original_annotations_preserved": True,
        "retrieval": "production pipeline live retrieval; no saved snapshot",
        "scoring": {
            "requested": "custom non-summary mixed F1: factoid/yesno token F1, list item F1; question macro-average",
            "list_parser": "remove whitespace then split on literal commas; preserve order and empty entries",
            "list_em": "ordered whole-list exact synonym matching after whitespace removal and lowercasing",
            "unresolved": "generation length and later official evaluation version",
            "official": "prepare separately; no official scoring performed",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
