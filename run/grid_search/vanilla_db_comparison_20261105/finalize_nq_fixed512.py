"""Append the requested sixth vanilla result without changing existing metrics."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "outputs/aaa_icde/re_vanilla"
CASE = ROOT / "06_nq_fixed512"


def main():
    results = [json.loads(line) for line in (CASE / "results.jsonl").open()]
    assert len(results) == 1
    result = results[0]
    assert result["status"] == "ok"
    assert result["TOP_K"] == "20" and result["EVAL_BSZ"] == "1"
    query_file = Path("/mnt/nvme1/datasets/dapr-nq-open/questions/query.jsonl")
    queries = [json.loads(line) for line in query_file.open() if line.strip()]
    predictions = next((CASE / "eval_outputs").glob("*.jsonl"))
    count = 0
    for line in predictions.open():
        if not line.strip():
            continue
        row = json.loads(line)
        assert row["id"] == queries[count]["id"]
        assert row["question"] == queries[count]["query"]
        count += 1
    assert count == len(queries) == result["count"] == 2390
    native_summary = next(CASE.glob("*summary2.csv"))
    with native_summary.open() as handle:
        native_rows = list(csv.DictReader(handle))
    assert len(native_rows) == 1
    native = native_rows[0]
    assert float(native["f1"]) == result["f1"]
    assert float(native["exact_match"]) == result["exact_match"]
    summary = ROOT / "summary2.csv"
    with summary.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    new_row = {
        "dataset": "NQ",
        "DB_TYPE": "fixed512",
        "EVAL_BSZ": "1",
        "DB_DIR": "/mnt/nvme1/datasets/dapr-nq-open/vanilla-bge-small-v1.5-512/db",
        **native,
    }
    existing = [
        row
        for row in rows
        if row["dataset"] == "NQ"
        and row["DB_TYPE"] == "fixed512"
        and row["EVAL_BSZ"] == "1"
        and row["TOP_K"] == "20"
    ]
    if existing:
        assert existing == [new_row]
    else:
        assert len(rows) == 5
        temporary = summary.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([*rows, new_row])
        temporary.replace(summary)
    completion = {
        "status": "complete",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "saved_predictions": count,
        "all_ids_and_questions_match_original_query_order": True,
        "native_summary2": str(native_summary),
        "combined_summary2": str(summary),
        "combined_summary2_row": new_row,
        "existing_five_rows_preserved": True,
    }
    (CASE / "completion_verified.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n"
    )
    launch_path = CASE / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["status"] = "complete"
    launch["completion_verified_file"] = str(CASE / "completion_verified.json")
    launch_path.write_text(json.dumps(launch, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
