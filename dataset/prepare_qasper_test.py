"""Index official QASPER test questions without transforming annotations."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    papers = json.loads(args.source.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            paper_id=pid,
            question_index=i,
            question_id=qa["question_id"],
            question=qa["question"],
        )
        for pid, paper in papers.items()
        for i, qa in enumerate(paper["qas"])
    ]
    assert len({r["question_id"] for r in rows}) == len(rows)
    for row in rows:
        original = papers[row["paper_id"]]["qas"][row["question_index"]]
        assert original["question_id"] == row["question_id"]
        assert original["question"] == row["question"]
    (args.output / "question_index.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    manifest = dict(
        source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        split="test",
        version="0.3",
        papers=len(papers),
        questions=len(rows),
        annotations=sum(len(qa["answers"]) for p in papers.values() for qa in p["qas"]),
        scope="Each question uses its original paired paper; no corpus retrieval",
        transformations="Question index only. All paper and QA fields remain in untouched source JSON. No filtering, answer flattening, text serialization, or evidence matching.",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
