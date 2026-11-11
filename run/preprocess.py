"""Build retrieval512, LateOn sentence100/region180, and sentence2bit artifacts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUBDIR = "sent-bge-small-v1.5-512-splitlong180"
FINAL_SUBDIR = "sent-bge-small-v1.5-512-splitlong100-lateon"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
LATEON = "lightonai/LateOn"
# Retrieval upsert/embedding batches from the existing construction records.
BATCHES = {
    "nq": (5461, 5461),
    "newsqa": (5461, 5461),
    "bioasq": (256, 128),
    "covid-qa": (256, 256),
}


def build_plan(args) -> list[dict]:
    root = args.dataset_root.resolve()
    source = args.source_db or root / SOURCE_SUBDIR / "db"
    source = source.resolve()
    final = root / FINAL_SUBDIR
    db_batch, embedding_batch = BATCHES[args.dataset]
    db_batch = args.db_batch_size or db_batch
    embedding_batch = args.embedding_batch_size or embedding_batch
    plan = []
    if args.stage in {"all", "retrieval"} and args.source_db is None:
        plan.append(
            {
                "name": "retrieval",
                "new_output": str(source),
                "command": [
                    sys.executable,
                    str(ROOT / "src/entrypoint/preprocess.py"),
                    "--docs_dir",
                    str(root / "documents"),
                    "--db_dir",
                    str(source),
                    "--model_name",
                    LLAMA,
                    "--retrievable_chunk_size",
                    "512",
                    "--max_subchunk_tokens",
                    "180",
                    "--materialize_db",
                    "True",
                    "--deduplicate_documents_by_hash",
                    "False",
                    "--db_batch_size",
                    str(db_batch),
                    "--chroma_embed_device",
                    "cuda",
                    "--chroma_embed_batch_size",
                    str(embedding_batch),
                ],
                "environment": {
                    "CHROMA_EMBED_BACKEND": "bge_small_v1_5",
                    "CHROMA_EMBED_DEVICE": "cuda",
                    "CHROMA_EMBED_BATCH_SIZE": str(embedding_batch),
                },
            }
        )
    if args.stage in {"all", "subchunks", "embeddings"}:
        plan.append(
            {
                "name": "download_lateon",
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); from encoder.lateon import LATEON_MODEL, LATEON_REVISION; from huggingface_hub import snapshot_download; snapshot_download(LATEON_MODEL, revision=LATEON_REVISION, allow_patterns=['*.json', '*.txt', '*.safetensors'])",
                    str(ROOT / "src"),
                ],
                "environment": {},
            }
        )
    if args.stage in {"all", "subchunks"}:
        plan.append(
            {
                "name": "subchunks",
                "new_output": str(final / "db"),
                "command": [
                    sys.executable,
                    str(ROOT / "run/build_subchunk_catalog.py"),
                    "--source-db",
                    str(source),
                    "--docs-dir",
                    str(root / "documents"),
                    "--output-db",
                    str(final / "db"),
                    "--log-dir",
                    str(args.log_dir / "subchunk_build"),
                    "--max-unit-tokens",
                    "100",
                    "--region-tokens",
                    "180",
                    "--workers",
                    str(args.workers),
                ],
                "environment": {},
            }
        )
    if args.stage in {"all", "embeddings"}:
        artifact = final / "colbert_window"
        plan.append(
            {
                "name": "embeddings",
                "new_output": str(artifact),
                "command": [
                    sys.executable,
                    str(ROOT / "src/entrypoint/materialize_subchunks.py"),
                    "--backend",
                    "colbert",
                    "--db_dir",
                    str(final / "db"),
                    "--docs_dir",
                    str(root / "documents"),
                    "--output_dir",
                    str(artifact),
                    "--model_name",
                    LATEON,
                    "--batch_size",
                    str(args.encoding_batch_size),
                    "--window_token_budget",
                    "180",
                    "--center_unit",
                    "subchunk_only",
                    "--embedding_storage",
                    "residual_2bit",
                    "--seed",
                    "123",
                    "--repo_path",
                    str(ROOT / "third_party/ColBERT"),
                ],
                "environment": {"COLBERT_MODEL_NAME": LATEON},
            }
        )
    return plan


def execute(plan, log_dir: Path) -> None:
    for step in plan:
        target = step.get("new_output")
        if target and Path(target).exists():
            raise FileExistsError(
                f"refusing to overwrite {step['name']} output: {target}"
            )
    for step in plan:
        target = step.get("new_output")
        if target and Path(target).exists():
            raise FileExistsError(
                f"refusing to overwrite {step['name']} output: {target}"
            )
        print(f"[{step['name']}] {shlex.join(step['command'])}", flush=True)
        env = os.environ.copy()
        env.update(step["environment"])
        env["PYTHONHASHSEED"] = "123"
        with (log_dir / f"{step['name']}.log").open("x") as log:
            result = subprocess.run(
                step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        status = {"stage": step["name"], "returncode": result.returncode}
        with (log_dir / "stages.jsonl").open("a") as handle:
            handle.write(json.dumps(status) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"{step['name']} failed; see {log_dir / (step['name'] + '.log')}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(BATCHES), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-db",
        type=Path,
        help="Explicitly reuse an existing idle retrieval DB instead of rebuilding it",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "retrieval", "subchunks", "embeddings"),
        default="all",
    )
    parser.add_argument("--db-batch-size", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--encoding-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "retrieval" and args.source_db:
        parser.error("--source-db is for reusing retrieval in the subchunks/all stages")
    for name in (
        "db_batch_size",
        "embedding_batch_size",
        "encoding_batch_size",
        "workers",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.log_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        args.log_dir = ROOT / "outputs/temp" / f"preprocess_{args.dataset}_{stamp}"
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(args)
    (args.log_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    )
    if args.dry_run:
        for step in plan:
            print(f"[{step['name']}] {shlex.join(step['command'])}")
        print(f"Plan: {args.log_dir / 'plan.json'}")
        return
    if not (args.dataset_root / "documents").is_dir():
        raise FileNotFoundError(args.dataset_root / "documents")
    execute(plan, args.log_dir)
    print(f"Completed; logs: {args.log_dir}")


if __name__ == "__main__":
    main()
