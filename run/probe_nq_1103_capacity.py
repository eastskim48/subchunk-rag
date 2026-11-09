"""Reuse the established longest-prompt, forced20-token GPU capacity probe."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import set_seed
    from model import LLMModel
    from probe_max_bsz_and_eval import (
        find_longest_prompt,
        run_forced_generation_probe,
        is_cuda_oom,
        clear_cuda_after_probe,
    )

    set_seed(123)
    torch.set_num_threads(32)
    llm = LLMModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=False,
        prompt_format="chat_system_user",
    )
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name, source in cfg["sources"].items():
        if args.condition and args.condition != name:
            continue
        if (out / f"{name}.json").exists():
            summary[name] = json.loads((out / f"{name}.json").read_text())
            continue
        prompt, maximum, average = find_longest_prompt(
            output_path=Path(source),
            tokenizer=llm.tokenizer,
            prompt_processor=llm.prompt_processor,
        )
        attempts = []

        def fits(batch):
            try:
                record = run_forced_generation_probe(
                    llm=llm, prompt=prompt, batch_size=batch, max_new_tokens=20
                )
                record["oom"] = False
                okay = True
            except Exception as error:
                if not is_cuda_oom(error):
                    raise
                clear_cuda_after_probe()
                record = {"batch_size": batch, "oom": True}
                okay = False
            attempts.append(record)
            with (out / "attempts.jsonl").open("a") as handle:
                handle.write(json.dumps({"condition": name, **record}) + "\n")
            print(name, record, flush=True)
            return okay

        low, high = 0, 1
        while fits(high):
            low, high = high, high * 2
        while high - low > 1:
            candidate = (low + high) // 2
            if fits(candidate):
                low = candidate
            else:
                high = candidate
        assert low > 0
        # Repeat the successful boundary after the failed allocation was cleared.
        assert fits(low)
        result = {
            "source": source,
            "maximum_input_tokens": maximum,
            "mean_input_tokens": average,
            "max_new_tokens": 20,
            "maximum_successful_batch": low,
            "smallest_failing_batch": high,
            "scope": "repeated longest saved actual chat prompt; forced20 output tokens; memory capacity, not maximum throughput",
            "allocator": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
            "attempts": attempts,
        }
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary[name] = result
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
