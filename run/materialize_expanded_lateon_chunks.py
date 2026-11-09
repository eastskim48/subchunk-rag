"""Build boundary-complete chunk artifacts using the existing LateOn encoder."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def wait_gpu():
    while True:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        other = [
            p.strip()
            for p in value.splitlines()
            if p.strip() and p.strip() != str(os.getpid())
        ]
        if not other:
            return
        print(f"GPU busy with {other}; waiting30s", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=["prepare", "build", "compress"], required=True
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    from materialize.expanded_chunk_materializer import build, finalize, prepare

    if args.stage == "prepare":
        print(json.dumps(prepare(config["db_dir"], config["work_dir"])), flush=True)
    else:
        wait_gpu()
        if args.stage == "build":
            build(config["db_dir"], config["work_dir"])
        else:
            finalize(config["work_dir"], config["output_dir"])


if __name__ == "__main__":
    main()
