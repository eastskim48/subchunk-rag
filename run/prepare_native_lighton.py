"""Prepare model-native artifacts without changing the evaluation pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def compress_nq(root):
    import faiss
    import numpy as np
    from materialize.colbert_residual import compress_colbert_window_artifact

    source = ROOT / "outputs/nq_lateon_fixed180_r010_chat_20261023/artifact_fp16"
    target = root / "nq/lateon_sentence_2bit"
    status = root / "nq/quantization_status.json"
    faiss.omp_set_num_threads(8)
    save(status, {"stage": "compressing", "source": str(source), "target": str(target)})
    if not target.exists():
        result = compress_colbert_window_artifact(
            source,
            target,
            seed=123,
            kmeans_niters=4,
            use_gpu=True,
            storage_layout="interleaved",
        )
        save(root / "nq/quantization_summary.json", result)
    assert np.array_equal(
        np.load(source / "data/offsets.npy"), np.load(target / "data/offsets.npy")
    )
    save(
        status, {"stage": "complete", "target": str(target), "offsets_identical": True}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--action", choices=["compress_nq"], required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    compress_nq(args.root.resolve())
