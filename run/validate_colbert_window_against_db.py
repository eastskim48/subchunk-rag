from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from materialize.colbert_window import (
    validate_colbert_window_artifact_against_db,
)  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python run/validate_colbert_window_against_db.py ARTIFACT_DIR DB_DIR"
        )
    artifact_dir, db_dir = sys.argv[1:]
    summary = validate_colbert_window_artifact_against_db(
        artifact_dir=artifact_dir,
        db_dir=db_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
