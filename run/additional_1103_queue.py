"""Run the user-specified additional grids after the active local8B experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def execute(cfg):
    out = Path(cfg["output"])
    failures = []
    for job in cfg["jobs"]:
        folder = Path(job["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / (job["name"] + ".queue_complete.json")
        if marker.exists():
            continue
        save(
            out / "status.json",
            {"stage": "running", "job": job["name"], "failures": failures},
        )
        try:
            for dependency in job.get("required", []):
                assert Path(dependency).exists(), dependency
            for command in job["commands"]:
                with (folder / (job["name"] + ".queue.log")).open("a") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
            if job.get("grid_results"):
                rows = [json.loads(line) for line in Path(job["grid_results"]).open()]
                assert rows and all(row.get("status") == "ok" for row in rows)
            save(marker, {"stage": "complete", "time": time.time()})
        except Exception as error:
            record = {
                "job": job["name"],
                "error_type": type(error).__name__,
                "message": str(error),
            }
            failures.append(record)
            save(folder / (job["name"] + ".queue_failed.json"), record)
    save(
        out / "status.json",
        {
            "stage": "complete_with_failures" if failures else "complete",
            "failures": failures,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if args.execute:
        execute(cfg)
        return
    while True:
        path = Path(cfg["preceding_status"])
        status = json.loads(path.read_text()) if path.exists() else {}
        if status.get("stage") in {"complete", "failed"}:
            break
        save(
            out / "status.json",
            {"stage": "waiting_for_preceding_experiment", "preceding": status},
        )
        time.sleep(30)
    while not all(Path(p).exists() for p in cfg["preparation_markers"]):
        save(out / "status.json", {"stage": "waiting_for_cpu_preparation"})
        time.sleep(30)
    command = [
        "bash",
        "run/run_grid.sh",
        "--no-exclusive",
        "--wait-lock",
        "--wait-gpu",
        "--interval",
        "30",
        "--lock-dir",
        str(Path.cwd() / ".runtime/gpu_locks"),
        "--",
        sys.executable,
        "-u",
        __file__,
        "--config",
        str(args.config),
        "--execute",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
