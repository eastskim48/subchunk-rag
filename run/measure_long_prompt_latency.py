"""Measure one-batch synthetic prompt latency using the established GPU probe."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=51200)
    parser.add_argument("--output-tokens", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    from model import LLMModel
    from probe_max_bsz_and_eval import run_forced_generation_probe

    # Claim the exclusive GPU before loading weights or preparing the prompt.
    reservation = torch.empty(1, device="cuda")
    torch.set_num_threads(32)
    transformers.set_seed(123)
    result = {
        "status": "running",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": "float16",
        "batch_size": 1,
        "requested_input_tokens": args.input_tokens,
        "forced_output_tokens": args.output_tokens,
        "prompt_format": "chat_system_user",
        "seed": 123,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "scope": "custom synthetic repeated-text performance measurement; no retrieval or quality evaluation",
        "measurements": [],
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        started = time.perf_counter()
        llm = LLMModel(
            model_name=result["model"],
            load_in_4bit=False,
            prompt_format="chat_system_user",
        )
        torch.cuda.synchronize()
        result["model_load_seconds"] = time.perf_counter() - started
        result["attention_implementation"] = llm.model.config._attn_implementation
        result["max_position_embeddings"] = llm.model.config.max_position_embeddings
        repeats = args.input_tokens
        for _ in range(10):
            prompt = llm.prompt_processor.build_qa_prompt(
                "What word is repeated in the context?", ["data " * repeats]
            )
            count = len(
                llm.tokenizer(prompt, add_special_tokens=True, truncation=False)[
                    "input_ids"
                ]
            )
            if count == args.input_tokens:
                break
            repeats += args.input_tokens - count
        else:
            raise RuntimeError("Could not construct exact-length synthetic prompt")
        truncated_count = len(
            llm.tokenizer(prompt, add_special_tokens=True, truncation=True)["input_ids"]
        )
        assert count == truncated_count == args.input_tokens
        (args.output / "prompt.txt").write_text(prompt)
        result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        result["actual_input_tokens"] = count
        save()
        short_prompt = llm.prompt_processor.build_qa_prompt(
            "What word is repeated?", ["data " * 128]
        )
        run_forced_generation_probe(
            llm=llm,
            prompt=short_prompt,
            batch_size=1,
            max_new_tokens=args.output_tokens,
        )
        original_generate = llm.model.generate
        phase_times = []

        def timed_generate(*positional, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_generate(*positional, **kwargs)
            torch.cuda.synchronize()
            phase_times.append(time.perf_counter() - start)
            return output

        llm.model.generate = timed_generate
        for trial in range(args.repeats):
            phase_times.clear()
            record = run_forced_generation_probe(
                llm=llm,
                prompt=prompt,
                batch_size=1,
                max_new_tokens=args.output_tokens,
            )
            assert record["prompt_tokens"] == args.input_tokens
            record.update(
                trial=trial + 1,
                prefill_seconds=phase_times[0],
                decode_seconds=sum(phase_times[1:]),
                inference_seconds=sum(phase_times),
            )
            result["measurements"].append(record)
            save()
            print(json.dumps(record), flush=True)
        result["median_inference_seconds"] = statistics.median(
            row["inference_seconds"] for row in result["measurements"]
        )
        result["timing_scope"] = (
            "CUDA-synchronized generate calls only; first token in prefill, remaining "
            "tokens in decode; excludes loading, CPU tokenization, device transfer, "
            "warmup and cache cleanup; no prefix cache reuse between trials"
        )
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        save()
        del reservation


if __name__ == "__main__":
    main()
