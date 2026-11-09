"""Time native XProvence batches on a fixed, real retrieved-context workload."""

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(cfg, out):
    import torch
    from transformers import set_seed
    from utils import parse_json_query_record
    from vectordb import ChromaDB

    torch.set_num_threads(cfg["cpu_threads"])
    set_seed(cfg["seed"])
    os.environ["CHROMA_EMBED_BACKEND"] = "bge_small_v1_5"
    os.environ["RETRIEVAL_INCLUDE_DOCUMENTS"] = "True"
    records = []
    with Path(cfg["query_file"]).open() as handle:
        for _ in range(cfg["query_count"]):
            records.append(parse_json_query_record(next(handle)))
    assert len({r["id"] for r in records}) == len(records)
    write_json(out / "status.json", {"phase": "retrieving", "queries": len(records)})
    db = ChromaDB(cfg["db_dir"])
    start = time.perf_counter()
    batches = db.find_top_k_docs(cfg["top_k"], [r["query"] for r in records])
    elapsed = time.perf_counter() - start
    assert len(batches) == len(records)
    payload = []
    for record, docs in zip(records, batches):
        assert len(docs) == cfg["top_k"]
        assert all(isinstance(d.text, str) and d.text.strip() for d in docs)
        payload.append(
            {**record, "documents": [{"id": d.id, "text": d.text} for d in docs]}
        )
    write_json(out / "retrieved_inputs.json", payload)
    write_json(
        out / "retrieval.json",
        {
            "seconds": elapsed,
            "source_query_file": cfg["query_file"],
            "db_dir": cfg["db_dir"],
            "query_ids": [r["id"] for r in records],
            "scope": "custom timing sample: first queries in original file order; no quality evaluation",
            "top_k": cfg["top_k"],
            "input_sha256": hashlib.sha256(
                (out / "retrieved_inputs.json").read_bytes()
            ).hexdigest(),
        },
    )
    write_json(out / "status.json", {"phase": "prepared"})
    print("prepared", len(records), "queries", flush=True)


def measure(cfg, out):
    import torch

    # Reserve the already-idle exclusive GPU before other imports or model loading.
    reservation = torch.empty(1, device="cuda")
    from transformers import set_seed
    import transformers
    from chunk import RetrievableChunk
    from model import LLMModel
    from compressor.methods.ml_selector import ProvenceCompressor

    torch.set_num_threads(cfg["cpu_threads"])
    set_seed(cfg["seed"])
    os.environ["PROVENCE_MODEL_NAME"] = cfg["xprovence_snapshot"]
    os.environ["PROVENCE_DEVICE"] = "cuda"
    os.environ["PROVENCE_THRESHOLD"] = str(cfg["threshold"])
    os.environ["PROVENCE_REORDER"] = "False"
    os.environ["PROVENCE_BATCH_SIZE"] = "256"
    state = {"phase": "loading", "attempts": []}
    write_json(out / "status.json", state)
    llm = LLMModel(
        cfg["llm_model"], load_in_4bit=False, prompt_format="chat_system_user"
    )
    compressor = ProvenceCompressor()
    assert not compressor.model.training
    rows = json.loads((out / "retrieved_inputs.json").read_text())
    queries = [r["query"] for r in rows]
    docs = [
        [RetrievableChunk(id=d["id"], text=d["text"]) for d in r["documents"]]
        for r in rows
    ]
    contexts = [[d.text for d in batch] for batch in docs]
    module = importlib.import_module(type(compressor.model).__module__)
    blocks = module.TestDataset(
        queries=queries,
        contexts=contexts,
        titles="first_sentence",
        tokenizer=compressor.model.tokenizer,
        max_len=compressor.model.max_len,
        enable_warnings=False,
    )
    lengths = [len(x) for x in blocks.ids]
    total_blocks = len(lengths)
    write_json(
        out / "workload.json",
        {
            "query_count": len(rows),
            "retrieved_chunk_occurrences": sum(map(len, docs)),
            "block_count": total_blocks,
            "block_lengths": lengths,
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": statistics.mean(lengths),
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "xprovence_dtype": str(next(compressor.model.parameters()).dtype),
            "xprovence_attention": compressor.model.config._attn_implementation,
            "llm_dtype": str(next(llm.model.parameters()).dtype),
            "llm_residency": "weights resident; no generation KV cache allocated",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "threshold": compressor.threshold,
            "reorder": compressor.reorder,
            "timing_scope": "native compression adapter including sentence splitting, tokenization, transfers, model and postprocessing; excludes retrieval/model loading; synchronized forward subtotal recorded separately",
        },
    )
    del blocks
    original_forward = compressor.model.forward
    forward_stats = []

    def timed_forward(*args, **kwargs):
        inputs = args[0] if args else kwargs["input_ids"]
        shape = list(inputs.shape)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        forward_stats.append({"shape": shape, "seconds": time.perf_counter() - start})
        return result

    compressor.model.forward = timed_forward
    # Small native warmup; measured batches start at 256 below.
    compressor.batch_size = 16
    compressor.compress_batch_top_k_docs([docs[0][:16]], [queries[0]])
    del forward_stats[:]
    baseline_digest = None

    def attempt(batch, repeats):
        nonlocal baseline_digest
        compressor.batch_size = batch
        successes = []
        for repeat in range(repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            forward_stats.clear()
            state.update(phase="measuring", batch_size=batch, repeat=repeat + 1)
            write_json(out / "status.json", state)
            record = {"batch_size": batch, "repeat": repeat + 1}
            start = time.perf_counter()
            try:
                selected = compressor.compress_batch_top_k_docs(docs, queries)
                torch.cuda.synchronize()
                record["compression_seconds"] = time.perf_counter() - start
                assert len(selected) == len(docs)
                assert [[d.id for d in ds] for ds in selected] == [
                    [d.id for d in ds] for ds in docs
                ]
                returned = [
                    [c.text for d in ds for c in d.cacheables] for ds in selected
                ]
                digest = hashlib.sha256(json.dumps(returned).encode()).hexdigest()
                if baseline_digest is None:
                    baseline_digest = digest
                record.update(
                    status="ok",
                    forward_seconds=sum(s["seconds"] for s in forward_stats),
                    forward_batches=list(forward_stats),
                    blocks_per_second=total_blocks / record["compression_seconds"],
                    queries_per_second=len(queries) / record["compression_seconds"],
                    output_sha256=digest,
                    output_matches_first_success=digest == baseline_digest,
                )
                del selected, returned
                successes.append(record)
            except torch.cuda.OutOfMemoryError as error:
                record.update(
                    status="oom",
                    elapsed_before_failure_seconds=time.perf_counter() - start,
                    error=str(error),
                    completed_forward_batches=list(forward_stats),
                )
            record.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            with (out / "attempts.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            state["attempts"].append(
                {k: v for k, v in record.items() if "forward_batches" not in k}
            )
            write_json(out / "status.json", state)
            print(json.dumps(state["attempts"][-1]), flush=True)
            if record["status"] == "oom":
                gc.collect()
                torch.cuda.empty_cache()
                return False
        return bool(successes)

    passed = 0
    failed = None
    batch = min(256, total_blocks)
    while True:
        if attempt(batch, cfg["repeats"]):
            passed = batch
            if batch == total_blocks:
                break
            batch = min(batch * 2, total_blocks)
        else:
            failed = batch
            break
    # Locate the capacity boundary at the explicitly recorded block granularity.
    resolution = cfg["capacity_resolution"]
    if failed is not None:
        while failed - passed > resolution:
            middle = ((passed + failed) // 2 // resolution) * resolution
            middle = max(resolution, middle)
            if middle <= passed or middle >= failed:
                break
            if attempt(middle, 1):
                passed = middle
            else:
                failed = middle
    assert passed > 0, "No successful batch size"
    # Repeat boundary points to avoid recommending a one-off timing result.
    successes_by_batch = {}
    for r in state["attempts"]:
        if r["status"] == "ok":
            successes_by_batch.setdefault(r["batch_size"], []).append(r)
    for candidate, records in sorted(successes_by_batch.items()):
        if len(records) < cfg["repeats"]:
            attempt(candidate, cfg["repeats"] - len(records))
    summaries = []
    for candidate in sorted(successes_by_batch):
        records = [
            r
            for r in state["attempts"]
            if r["batch_size"] == candidate and r["status"] == "ok"
        ]
        median = statistics.median(r["compression_seconds"] for r in records)
        summaries.append(
            {
                "batch_size": candidate,
                "successful_repeats": len(records),
                "median_compression_seconds": median,
                "median_forward_seconds": statistics.median(
                    r["forward_seconds"] for r in records
                ),
                "queries_per_second": len(rows) / median,
                "blocks_per_second": total_blocks / median,
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
                "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in records),
                "all_outputs_match_first_success": all(
                    r["output_matches_first_success"] for r in records
                ),
            }
        )
    result = {
        "status": "complete",
        "conditions": cfg,
        "measurements": summaries,
        "fastest_tested_batch": min(
            summaries, key=lambda r: r["median_compression_seconds"]
        )["batch_size"],
        "largest_successful_batch": passed,
        "smallest_failing_batch": failed,
        "capacity_resolution": resolution,
        "capacity_limited_by_workload": failed is None,
        "scope": "measured timing/capacity for this fixed custom workload and resident model weights; not a guarantee for longer inputs or generation KV-cache coexistence",
    }
    write_json(out / "result.json", result)
    state["phase"] = "complete"
    write_json(out / "status.json", state)
    del reservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "measure"], required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "prepare":
            prepare(cfg, out)
        else:
            measure(cfg, out)
    except Exception as error:
        write_json(out / "failure.json", {"error": repr(error), "phase": args.phase})
        raise


if __name__ == "__main__":
    main()
