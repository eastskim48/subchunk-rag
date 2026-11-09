"""Run the native evaluator with direct region search and budgeted assembly."""

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    # Keep the exclusive GPU context while the CPU retrieval index is loaded.
    while True:
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        if any(pid.strip() and pid.strip() != str(os.getpid()) for pid in pids):
            print("waiting_for_gpu", pids, flush=True)
            time.sleep(30)
            continue
        try:
            reservation = torch.empty(1, device="cuda")
            break
        except RuntimeError as error:
            if "busy or unavailable" not in str(error):
                raise
            print("waiting_for_cuda_context", flush=True)
            time.sleep(30)

    from direct_region_vectordb import DirectRegionVectorDB
    from entrypoint import eval as entrypoint

    output = Path(os.environ["OUTPUT_FILE"])
    output.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.ChromaDB = lambda db_dir: DirectRegionVectorDB(
        db_dir,
        source_db=os.environ["DIRECT_SOURCE_DB"],
        trace_path=str(output) + ".retrieval.jsonl",
        query_batch_size=int(os.environ["EVAL_BSZ"]),
    )
    entrypoint.main(
        query_file=os.environ["QUERY_FILE"],
        answer_file=os.environ["ANSWER_FILE"],
        db_dir=os.environ["DB_DIR"],
        cache_dir=os.environ["CACHE_DIR"],
        dataset=os.environ["EVAL_DATASET"],
        top_k=int(os.environ["DIRECT_INITIAL_SEARCH_DEPTH"]),
        use_past_cache=False,
        bsz=int(os.environ["EVAL_BSZ"]),
        max_new_tokens=int(os.environ["MAX_NEW_TOKENS"]),
        output_file=str(output),
        model_name=os.environ["MODEL_NAME"],
        compress_method=None,
        use_cleaner=False,
        model_load_in_4bit=False,
        prompt_format=os.environ["PROMPT_FORMAT"],
        seed=int(os.environ["EVAL_SEED"]),
        cpu_threads=int(os.environ["EVAL_CPU_THREADS"]),
    )
    del reservation


if __name__ == "__main__":
    main()
