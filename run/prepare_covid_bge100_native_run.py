"""Prepare the explicitly requested COVID BGE Top100/window100 pair."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/aaa_icde/covid_bge100_window100_r002_generation_20261029"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/mnt/nvme1/datasets/covid-qa-full-test")
BASE = DATA / "sent-bge-small-v1.5-512-splitlong100-lateon"
VIEW = DATA / "lateon_sentence100_native_window_sweep_20261029/window100"
old = json.loads(
    (
        ROOT / "outputs/aaa_icde/covid_lateon100_window_sweep_20261029/w100/config.json"
    ).read_text()
)
jobs = []
for method, compressor in [
    ("denseon", "dense_sliding_subchunk"),
    ("lateon", "colbert_sliding_region"),
]:
    cfg = {
        k: old[k]
        for k in [
            "source_dataset",
            "dataset",
            "dataset_description",
            "queries",
            "answers",
            "count",
        ]
    }
    cfg.update(
        output=str(OUT),
        method=method,
        ratio=0.02,
        native_window=100,
        source=str(DATA / "test.jsonl"),
        positions=str(
            ROOT
            / "outputs/aaa_icde/covid_sentence100_answer_char_recall_20261029/sentence_positions.json"
        ),
        dense_artifact=str(BASE / "denseon_region100_fp16"),
    )
    cfg["env"] = old["env"] | {
        "DB_DIR": str(VIEW / "db"),
        "COLBERT_WINDOW_DIR": str(VIEW / "colbert_lateon_sentence_2bit"),
        "TOP_K": "100",
        "RETAIN_TOKEN_RATIO": "0.02",
        "DENSE_EMBED_MODEL": "lightonai/DenseOn",
        "DENSE_EMBED_DEVICE": "cpu",
        "DENSE_EMBED_BATCH_SIZE": "24",
    }
    folder = OUT / method
    folder.mkdir(exist_ok=True)
    cfg["methods"] = {
        method: {
            "COMPRESS_METHOD": compressor,
            "OUTPUT_FILE": str(folder / "predictions.jsonl"),
        }
    }
    paths = [Path(cfg[k]) for k in ["queries", "answers", "source", "positions"]]
    paths += [
        VIEW / "db/build_manifest.json",
        VIEW / "db/subchunks.json",
        VIEW / "colbert_lateon_sentence_2bit/index.json",
        BASE / "denseon_region100_fp16/regions.sqlite3",
        BASE / "denseon_region100_fp16/dense_vectors.npy",
    ]
    paths += [
        ROOT / p
        for p in [
            "test/run_covid_native_recall_generation.py",
            "test/run_hotpot_lighton_suite.py",
            "test/run_saved_prompt_comparison.py",
            "src/engine.py",
            "src/acc_metric.py",
            "src/compressor/methods/dense.py",
            "src/compressor/token_budget.py",
        ]
    ]
    cfg["input_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    cfg["notes"] = (
        "Custom COVID original147 documents/all2019 queries. Native BGE Top100 -> fixed100/r.02 -> Llama8B per batch24. Latest LateOn100 splitter. Runtime src budget, no override, no pruning. Existing DenseOn region vectors and sentence LateOn2bit reused. Original answer character union recall recorded on actual selected source IDs."
    )
    config_path = OUT / f"{method}.config.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    jobs.append(
        dict(
            name=method,
            command=[
                "/home/dongseob/miniconda3/bin/python",
                "-u",
                "test/run_covid_native_recall_generation.py",
                "--config",
                str(config_path),
            ],
            log=str(folder / "run.log"),
        )
    )
queue = dict(
    output=str(OUT),
    cwd=str(ROOT),
    poll_seconds=30,
    env={"PYTHONUNBUFFERED": "1"},
    jobs=jobs,
)
(OUT / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")
print(OUT)
