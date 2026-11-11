"""CLI entrypoint for retrieval, compression, and answer evaluation."""

import fire
import time
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
from transformers import logging, set_seed
from engine import QueryProcessor
from vectordb import ChromaDB
from model import LLMModel
from metric.acc_metric import evaluate
from typing import Optional


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def main(
    query_file: str,
    db_dir: str,
    dataset: Optional[str] = None,
    top_k: int = 4,
    bsz: int = 1,
    max_new_tokens: int = 20,
    total_num: Optional[int] = None,
    output_file: str = "log.jsonl",
    answer_file: str = "answer.jsonl",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    compress_method: Optional[str] = None,
    use_cleaner: bool = False,
    model_load_in_4bit: bool = False,
    prompt_format: str = "chat_system_user",
    seed: int = 123,
    cpu_threads: int = 32,
):
    use_cleaner = _coerce_bool(use_cleaner)
    model_load_in_4bit = _coerce_bool(model_load_in_4bit)

    if Path(output_file).exists():
        raise FileExistsError(output_file)
    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    total_start = time.perf_counter()
    logging.set_verbosity_error()
    init_start = time.perf_counter()
    vectordb = ChromaDB(db_dir=db_dir)
    processor = QueryProcessor(
        query_file=query_file,
        vectordb=vectordb,
        model=LLMModel(
            model_name=model_name,
            load_in_4bit=model_load_in_4bit,
            prompt_format=prompt_format,
        ),
        top_k=top_k,
        output_file=output_file,
        compress_method=compress_method,
    )
    init_runtime = time.perf_counter() - init_start
    run_stats = processor.process_query(
        bsz=bsz, max_new_tokens=max_new_tokens, total_num=total_num
    )
    setup_runtime = 0.0
    run_runtime = 0.0
    process_total_runtime = 0.0
    if run_stats is not None:
        setup_runtime = run_stats.get("setup_time", 0.0)
        run_runtime = run_stats.get("run_time", 0.0)
        process_total_runtime = run_stats.get(
            "process_total_time", setup_runtime + run_runtime
        )
        processed_queries = run_stats.get("processed_queries", 0)
        processed_batches = run_stats.get("processed_batches", 0)
        warmup_time = run_stats.get("warmup_time", 0.0)
        warmup_queries = run_stats.get("warmup_queries", 0)
        if run_runtime > 0 and processed_queries > 0:
            print(f"throughput | requests/sec: {processed_queries / run_runtime:.4f}")
        if run_runtime > 0 and processed_batches > 0:
            print(f"throughput | batches/sec: {processed_batches / run_runtime:.4f}")
        if warmup_queries > 0:
            print(
                f"warmup | queries: {warmup_queries}, batch_size: 1, time: {warmup_time:.4f} seconds"
            )
    score_start = time.perf_counter()
    if dataset == "bioasq":
        from metric.bioasq_metrics import evaluate_saved_answers

        evaluate_saved_answers(output_file, answer_file, query_file)
    else:
        evaluate(
            prediction_file=output_file,
            ground_truth_file=answer_file,
            dataset=dataset,
            use_cleaner=use_cleaner,
        )
    score_runtime = time.perf_counter() - score_start
    total_runtime = process_total_runtime + score_runtime
    end_to_end_runtime = time.perf_counter() - total_start
    print(f"init time: {init_runtime:.4f} seconds")
    print(f"setup time: {setup_runtime:.4f} seconds")
    print(f"run time: {run_runtime:.4f} seconds")
    print(f"process total time: {process_total_runtime:.4f} seconds")
    print(f"score time: {score_runtime:.4f} seconds")
    print(f"TOTAL: {total_runtime:.4f} seconds")
    print(f"end-to-end time: {end_to_end_runtime:.4f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
