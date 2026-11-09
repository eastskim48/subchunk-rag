"""Run an explicit experiment list, waiting for GPU 0 and advancing on failure."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_pids():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def run(config):
    root = Path(config["output"])
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root / "status.json"
    events = root / "events.jsonl"
    results = {}

    def record(**state):
        value = dict(updated=time.time(), results=results.copy(), **state)
        save(status, value)
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    for job in config["jobs"]:
        name = job["name"]
        marker = root / (name + ".terminal.json")
        if marker.exists():
            results[name] = json.loads(marker.read_text())
            continue
        try:
            command = job["command"]
            if "--config" in command:
                config_path = Path(command[command.index("--config") + 1])
                if not config_path.is_absolute():
                    config_path = Path(config["cwd"]) / config_path
                job_config = json.loads(config_path.read_text())
                if "jina" in json.dumps(job_config).lower():
                    raise ValueError("Retired model/tokenizer dependency in job config")
                if "CONTRIBUTOR_WINDOW_TOKENIZER" in job_config.get("env", {}):
                    raise ValueError(
                        "Region tokenizer overrides are no longer supported"
                    )
        except Exception as exc:
            value = dict(returncode=125, state="invalid_config", error=repr(exc))
            save(marker, value)
            results[name] = value
            record(stage="job_skipped", job=name, **value)
            continue
        failed_dependencies = [
            d for d in job.get("requires", []) if results[d]["returncode"] != 0
        ]
        if failed_dependencies:
            value = dict(
                returncode=125,
                state="blocked_by_failed_dependency",
                dependencies=failed_dependencies,
            )
            save(marker, value)
            results[name] = value
            record(stage="job_skipped", job=name, **value)
            continue
        while True:
            try:
                active = gpu_pids()
            except Exception as exc:
                record(stage="gpu_poll_error", job=name, error=repr(exc))
                time.sleep(config["poll_seconds"])
                continue
            if not active:
                break
            record(stage="waiting_for_gpu", job=name, gpu_pids=active)
            time.sleep(config["poll_seconds"])
        log = Path(job["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            with log.open("a") as stream:
                child = subprocess.Popen(
                    job["command"],
                    cwd=config["cwd"],
                    env=os.environ | config["env"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                record(
                    stage="running",
                    job=name,
                    pid=child.pid,
                    log=str(log),
                    started=started,
                )
                code = child.wait()
            value = dict(
                returncode=code,
                state="complete" if code == 0 else "failed",
                started=started,
                finished=time.time(),
                log=str(log),
            )
        except Exception as exc:
            value = dict(
                returncode=126,
                state="failed_to_launch",
                error=repr(exc),
                finished=time.time(),
            )
        save(marker, value)
        results[name] = value
        record(stage="job_finished", job=name, **value)
    record(
        stage="finished",
        failed_jobs=[
            name for name, result in results.items() if result["returncode"] != 0
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
