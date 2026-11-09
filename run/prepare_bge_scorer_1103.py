"""Reuse native region definitions and existing materialization for BGE-small."""

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["snapshot", "embed"], required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CHROMA_EMBED_BACKEND="bge_small_v1_5",
        CHROMA_EMBED_DEVICE="cpu",
    )
    import torch
    from transformers import set_seed
    from run_nq_materialized_dense_regions import (
        build_snapshot,
        materialize_regions,
        load_snapshot,
    )

    torch.set_num_threads(8)
    set_seed(123)
    if args.phase == "embed":
        # Hold this process's exclusive CUDA context during CPU region preparation.
        # Otherwise another GPU job can start before the model is moved to CUDA.
        while True:
            pids = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
            if any(pid.strip() and pid.strip() != str(os.getpid()) for pid in pids):
                print("waiting_for_gpu_before_embedding", pids, flush=True)
                time.sleep(30)
                continue
            try:
                torch.cuda.init()
                gpu_reservation = torch.empty(1, device="cuda")
                break
            except RuntimeError as error:
                if "busy or unavailable" not in str(error):
                    raise
                print("waiting_for_exclusive_cuda_context", flush=True)
                time.sleep(30)
    for name, spec in cfg["datasets"].items():
        if args.dataset and args.dataset != name:
            continue
        print(args.phase, name, flush=True)
        snapshot = Path(spec["snapshot"])
        if args.phase == "snapshot":
            if Path(f"{snapshot}.index.json").exists():
                load_snapshot(snapshot)
                continue
            build_snapshot(
                SimpleNamespace(
                    query_file=spec["queries"],
                    output=str(snapshot),
                    db_dir=spec["db"],
                    top_k=100,
                    batch_size=28,
                    dataset=spec["dataset"],
                )
            )
        else:
            output = Path(spec["artifact"])
            if (output / "index.json").exists():
                index = json.loads((output / "index.json").read_text())
                assert index["embedding_model"] == "BAAI/bge-small-en-v1.5"
                continue
            materialize_regions(
                SimpleNamespace(
                    snapshot=str(snapshot),
                    output_dir=str(output),
                    colbert_window_dir=spec["sentence_artifact"],
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cuda",
                    batch_size=1024,
                )
            )


if __name__ == "__main__":
    main()
