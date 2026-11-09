"""Prepare all full-corpus region lists, encode under the GPU lock, then index."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_bge_direct_regions import read, save


def run_phase(cfg, config_path, phase):
    failures = []
    log_dir = Path(cfg["logs"])
    for name, spec in cfg["datasets"].items():
        out = Path(spec["output"])
        required = {
            "prepare": None,
            "encode": "prepared.json",
            "index": "encoded.json",
        }[phase]
        if required and not (out / required).exists():
            failures.append(
                dict(dataset=name, phase=phase, error=f"Missing {required}")
            )
            continue
        save(
            log_dir / "status.json", dict(stage=phase, dataset=name, failures=failures)
        )
        with (log_dir / f"{name}.{phase}.log").open("a") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "run/build_bge_direct_regions.py",
                    "--config",
                    str(config_path),
                    "--dataset",
                    name,
                    "--phase",
                    phase,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode:
            failures.append(dict(dataset=name, phase=phase, returncode=proc.returncode))
    save(log_dir / f"{phase}_result.json", dict(failures=failures))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config)
    log_dir = Path(cfg["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":" + os.environ["PATH"]
    if args.encode_only:
        run_phase(cfg, args.config, "encode")
        return
    if args.index_only:
        started = time.time()
        failures = run_phase(cfg, args.config, "index")
        save(
            log_dir / "status.json",
            dict(
                stage="complete_with_failures" if failures else "complete",
                failures=failures,
                started=started,
                finished=time.time(),
            ),
        )
        return
    started = time.time()
    failures = run_phase(cfg, args.config, "prepare")
    save(log_dir / "status.json", dict(stage="waiting_for_gpu", failures=failures))
    proc = subprocess.run(
        [
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
            "--encode-only",
        ]
    )
    if proc.returncode:
        failures.append(dict(phase="gpu_wrapper", returncode=proc.returncode))
    if (log_dir / "encode_result.json").exists():
        failures.extend(read(log_dir / "encode_result.json")["failures"])
    failures.extend(run_phase(cfg, args.config, "index"))
    save(
        log_dir / "status.json",
        dict(
            stage="complete_with_failures" if failures else "complete",
            failures=failures,
            started=started,
            finished=time.time(),
        ),
    )


if __name__ == "__main__":
    main()
