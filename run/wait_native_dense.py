"""Wait for the explicit CPU region build, then run DenseOn materialization."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
cfg = json.load(a.config.open())
folder = Path(cfg["artifacts"])
while not (folder / "regions_complete.json").exists():
    state = subprocess.run(
        ["systemctl", "--user", "is-active", cfg["region_service"]],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if state not in {"active", "activating"}:
        raise RuntimeError(f"Region build stopped before completion: {state}")
    print("Waiting for native region definitions", flush=True)
    time.sleep(30)
subprocess.run(
    [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("build_native180_regions.py")),
        "--config",
        str(a.config),
        "--action",
        "dense",
    ],
    check=True,
)
