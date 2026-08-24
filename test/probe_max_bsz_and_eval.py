"""Probe LLM max batch size with the longest saved prompt, then run grids.

This is a one-off throughput utility. It does not change the shared evaluation
or generation implementation. Multiple grids run sequentially in isolated
child processes so the outer GPU lock and exclusive mode remain active once.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn

import torch
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import LLMModel  # noqa: E402
from prompt import PromptProcessor  # noqa: E402

PROBE_CASE_KEYS = {"PROBE_INITIAL_BSZ"}
IGNORED_MATCH_KEYS = {"EVAL_BSZ", "OUTPUT_FILE", *PROBE_CASE_KEYS}


@dataclass
class CasePlan:
    index: int
    case: dict[str, str]
    source_result: dict[str, Any]
    source_output: Path
    longest_prompt: str
    max_prompt_tokens: int
    mean_prompt_tokens: float
    initial_bsz: int
    selected_bsz: int | None = None


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return stringify(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def iter_eval_cases(config: dict[str, Any]) -> Iterable[dict[str, str]]:
    eval_cases = config.get("eval_cases") or []
    eval_axes = config.get("eval_axes") or {}
    if eval_cases and eval_axes:
        raise ValueError("eval_cases and eval_axes are mutually exclusive")
    if eval_cases:
        for case in eval_cases:
            if not isinstance(case, dict):
                raise ValueError("each eval_cases entry must be a mapping")
            yield {key: stringify(value) for key, value in case.items()}
        return
    if not eval_axes:
        raise ValueError("the grid must define eval_cases or eval_axes")
    axes = list(eval_axes)
    values = [
        value if isinstance(value, list) else [value]
        for value in (eval_axes[key] for key in axes)
    ]
    for combination in itertools.product(*values):
        yield {
            key: stringify(value) for key, value in zip(axes, combination, strict=True)
        }


def choose_initial_bsz(
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    target_padded_tokens: int,
    step: int,
    min_bsz: int,
    max_bsz: int,
    override: str = "",
) -> int:
    if step <= 0 or min_bsz <= 0 or max_bsz < min_bsz:
        raise ValueError("invalid batch-size min/max/step")
    if override:
        candidate = int(override)
    else:
        tokens_per_request = max_prompt_tokens + max_new_tokens
        candidate = target_padded_tokens // tokens_per_request
        candidate = (candidate // step) * step
    candidate = min(candidate, max_bsz)
    if candidate < min_bsz:
        candidate = min_bsz
    if candidate % step != 0:
        raise ValueError(
            f"initial batch size {candidate} is not divisible by step {step}"
        )
    return candidate


def case_matches_result(case: dict[str, str], result: dict[str, Any]) -> bool:
    for key, value in case.items():
        if key in IGNORED_MATCH_KEYS:
            continue
        if stringify(result.get(key)) != stringify(value):
            return False
    return True


def resolve_source_output(source_run_dir: Path, output_value: str) -> Path:
    configured = Path(output_value)
    if configured.exists():
        return configured
    local = source_run_dir / "eval_outputs" / configured.name
    if local.exists():
        return local
    alternatives = list(
        (source_run_dir / "eval_outputs").glob(
            configured.name.replace("__compress_method=__", "__compress_method=none__")
        )
    )
    if len(alternatives) == 1:
        return alternatives[0]
    raise FileNotFoundError(
        f"cannot resolve source OUTPUT_FILE={output_value!r} under {source_run_dir}"
    )


def select_source_result(
    *,
    source_results: list[dict[str, Any]],
    preprocess_name: str,
    case: dict[str, str],
) -> dict[str, Any]:
    matches = [
        result
        for result in source_results
        if result.get("status") == "ok"
        and stringify(result.get("preprocess_name")) == preprocess_name
        and case_matches_result(case, result)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one source result for case={case}, got {len(matches)}"
        )
    return matches[0]


def find_longest_prompt(
    *,
    output_path: Path,
    tokenizer,
    prompt_processor: PromptProcessor,
    batch_size: int = 256,
) -> tuple[str, int, float]:
    longest_prompt = ""
    max_tokens = -1
    total_tokens = 0
    count = 0
    batch_prompts: list[str] = []

    def consume(prompts: list[str]) -> None:
        nonlocal longest_prompt, max_tokens, total_tokens, count
        token_ids = tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )["input_ids"]
        for prompt, ids in zip(prompts, token_ids, strict=True):
            length = len(ids)
            total_tokens += length
            count += 1
            if length > max_tokens:
                max_tokens = length
                longest_prompt = prompt

    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question")
            contexts = record.get("ctxs")
            if not isinstance(question, str) or not isinstance(contexts, list):
                raise ValueError(
                    f"{output_path}:{line_number} lacks question/ctxs fields"
                )
            passages = []
            for context in contexts:
                if not isinstance(context, dict) or not isinstance(
                    context.get("text"), str
                ):
                    raise ValueError(
                        f"{output_path}:{line_number} has an invalid ctxs entry"
                    )
                passages.append(context["text"])
            batch_prompts.append(
                prompt_processor.build_cache_aligned_qa_prompt(
                    query=question, passages=passages
                )
            )
            if len(batch_prompts) == batch_size:
                consume(batch_prompts)
                batch_prompts = []
    if batch_prompts:
        consume(batch_prompts)
    if count == 0:
        raise ValueError(f"source output is empty: {output_path}")
    return longest_prompt, max_tokens, total_tokens / count


def is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    lowered = str(error).lower()
    return "out of memory" in lowered or "cublas_status_alloc_failed" in lowered


def clear_cuda_after_probe() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_forced_generation_probe(
    *,
    llm: LLMModel,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    tokenizer = llm.tokenizer
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"].repeat(batch_size, 1)
    attention_mask = encoded["attention_mask"].repeat(batch_size, 1)
    prompt_tokens = int(input_ids.shape[1])

    first_output = None
    final_output = None
    tokens = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    try:
        tokens = {
            "input_ids": input_ids.to("cuda"),
            "attention_mask": attention_mask.to("cuda"),
        }
        with torch.inference_mode():
            first_output = llm.model.generate(
                **tokens,
                min_new_tokens=1,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=True,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            if max_new_tokens == 1:
                final_output = first_output
            else:
                continued_attention_mask = (
                    first_output.sequences != tokenizer.pad_token_id
                ).long()
                final_output = llm.model.generate(
                    input_ids=first_output.sequences,
                    attention_mask=continued_attention_mask,
                    min_new_tokens=max_new_tokens - 1,
                    max_new_tokens=max_new_tokens - 1,
                    use_cache=True,
                    past_key_values=first_output.past_key_values,
                    eos_token_id=tokenizer.pad_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
        torch.cuda.synchronize()
        generated_tokens = int(final_output.sequences.shape[1] - prompt_tokens)
        if generated_tokens != max_new_tokens:
            raise RuntimeError(
                f"dummy generation produced {generated_tokens} tokens, "
                f"expected {max_new_tokens}"
            )
        return {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_before_bytes": int(free_before),
            "total_gpu_memory_bytes": int(total_memory),
        }
    finally:
        del final_output
        del first_output
        del tokens
        del input_ids
        del attention_mask
        clear_cuda_after_probe()


def make_selected_grid(config: dict[str, Any], plans: list[CasePlan]) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    selected.pop("max_prompt_bsz_probe", None)
    selected.pop("auto_eval_bsz", None)
    selected.pop("eval_axes", None)
    selected_cases = []
    for plan in plans:
        if plan.selected_bsz is None:
            raise ValueError(f"case {plan.index} has no selected batch size")
        case = {
            key: value for key, value in plan.case.items() if key not in PROBE_CASE_KEYS
        }
        case["EVAL_BSZ"] = str(plan.selected_bsz)
        selected_cases.append(case)
    selected["eval_cases"] = selected_cases
    selected.setdefault("fixed_env", {})["EVAL_BSZ"] = "1"
    return selected


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def restore_completed_probes(
    path: Path,
    plans: list[CasePlan],
    *,
    max_new_tokens: int,
    target_padded_tokens: int,
    step: int,
    min_bsz: int,
    max_bsz: int,
) -> int:
    if not path.exists():
        return 0
    records = load_jsonl(path)
    indices = [record.get("case_index") for record in records]
    if indices != list(range(len(records))):
        raise ValueError(
            f"probe resume records must be a unique ordered prefix, got {indices}"
        )
    if len(records) > len(plans):
        raise ValueError(
            f"probe resume has {len(records)} records for only {len(plans)} cases"
        )

    for record, plan in zip(records, plans, strict=False):
        expected = {
            "case_index": plan.index,
            "case": plan.case,
            "source_output": str(plan.source_output),
            "max_prompt_tokens": plan.max_prompt_tokens,
            "mean_prompt_tokens": plan.mean_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "target_padded_tokens": target_padded_tokens,
            "initial_bsz": plan.initial_bsz,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(
                    f"probe resume mismatch for case {plan.index} field {key}: "
                    f"saved={record.get(key)!r}, current={value!r}"
                )
        optional_probe_config = {
            "step": step,
            "min_bsz": min_bsz,
            "max_bsz": max_bsz,
        }
        for key, value in optional_probe_config.items():
            if key in record and record[key] != value:
                raise ValueError(
                    f"probe resume mismatch for case {plan.index} field {key}: "
                    f"saved={record[key]!r}, current={value!r}"
                )

        selected_bsz = record.get("selected_bsz")
        if not isinstance(selected_bsz, int) or isinstance(selected_bsz, bool):
            raise ValueError(
                f"probe resume case {plan.index} has invalid selected_bsz "
                f"{selected_bsz!r}"
            )
        if not min_bsz <= selected_bsz <= plan.initial_bsz:
            raise ValueError(
                f"probe resume case {plan.index} selected_bsz={selected_bsz} "
                f"is outside [{min_bsz}, {plan.initial_bsz}]"
            )
        if selected_bsz % step != 0:
            raise ValueError(
                f"probe resume case {plan.index} selected_bsz={selected_bsz} "
                f"is not divisible by step={step}"
            )
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not any(
            isinstance(attempt, dict)
            and attempt.get("oom") is False
            and attempt.get("bsz") == selected_bsz
            for attempt in attempts
        ):
            raise ValueError(
                f"probe resume case {plan.index} lacks a successful selected attempt"
            )
        plan.selected_bsz = selected_bsz
        print(
            f"[probe-resume] case={plan.index} selected_bsz={selected_bsz}",
            flush=True,
        )
    return len(records)


def write_or_validate_selected_grid(path: Path, selected_grid: dict[str, Any]) -> None:
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != selected_grid:
            raise ValueError(
                f"existing selected grid does not match current plan: {path}"
            )
        print(f"[selected-grid-resume] {path}", flush=True)
        return
    path.write_text(yaml.safe_dump(selected_grid, sort_keys=False), encoding="utf-8")
    print(f"[selected-grid] {path}", flush=True)


def replace_process_with_grid_eval(selected_grid_path: Path) -> NoReturn:
    command = [
        sys.executable,
        str(REPO_ROOT / "run/grid_search/eval.py"),
        str(selected_grid_path),
    ]
    print(f"[eval] {' '.join(command)}", flush=True)
    os.chdir(REPO_ROOT)
    os.execve(sys.executable, command, os.environ.copy())
    raise RuntimeError("os.execve returned unexpectedly")


def run_grids_sequentially(
    grids: list[Path], *, dry_run: bool, probe_only: bool
) -> int:
    base_command = [sys.executable, str(Path(__file__).resolve())]
    if dry_run:
        base_command.append("--dry-run")
    if probe_only:
        base_command.append("--probe-only")
    overall_status = 0
    for index, grid in enumerate(grids, start=1):
        command = [*base_command, str(grid)]
        print(
            f"[grid-sequence] start={index}/{len(grids)} grid={grid}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy())
        if completed.returncode != 0:
            overall_status = 1
            print(
                f"[grid-sequence] failed={index}/{len(grids)} "
                f"grid={grid} status={completed.returncode}",
                flush=True,
            )
        else:
            print(
                f"[grid-sequence] done={index}/{len(grids)} grid={grid}",
                flush=True,
            )
    return overall_status


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    probe = config.get("max_prompt_bsz_probe")
    if not isinstance(probe, dict):
        raise ValueError("grid must define max_prompt_bsz_probe")
    groups = config.get("eval_groups") or []
    if len(groups) != 1:
        raise ValueError("this utility requires exactly one eval_group")
    preprocess_name = stringify(groups[0].get("name"))
    if not preprocess_name:
        raise ValueError("eval_groups[0].name is required")
    fixed_env = config.get("fixed_env") or {}
    eval_fixed_env = config.get("eval_fixed_env") or {}
    prompt_format = stringify(fixed_env.get("PROMPT_FORMAT", "raw_chunk_first"))
    if prompt_format != "raw_chunk_first":
        raise ValueError("max-prompt probing currently requires raw_chunk_first")
    use_past_cache = stringify(eval_fixed_env.get("EVAL_USE_PAST_CACHE", "False"))
    if coerce_bool(use_past_cache):
        raise ValueError("max-prompt probing is cache-off only")
    return probe, preprocess_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("grids", type=Path, nargs="+")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute exact longest prompts and initial candidates without GPU probing",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="probe and write the selected grid without launching full evaluation",
    )
    args = parser.parse_args()

    if len(args.grids) > 1:
        return run_grids_sequentially(
            args.grids, dry_run=args.dry_run, probe_only=args.probe_only
        )

    grid_path = args.grids[0].resolve()
    config = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("grid YAML root must be a mapping")
    probe, preprocess_name = validate_config(config)

    source_run_dir = Path(probe["source_run_dir"])
    if not source_run_dir.is_absolute():
        source_run_dir = REPO_ROOT / source_run_dir
    source_manifest = json.loads(
        (source_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if stringify(source_manifest.get("dataset")) != stringify(config.get("dataset")):
        raise ValueError("source run dataset does not match probe grid dataset")
    source_results = load_jsonl(source_run_dir / "results.jsonl")

    fixed_env = config.get("fixed_env") or {}
    model_name = stringify(
        fixed_env.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    )
    max_new_tokens = int(
        probe.get("max_new_tokens", fixed_env.get("MAX_NEW_TOKENS", 20))
    )
    target_padded_tokens = int(probe["target_padded_tokens"])
    step = int(probe.get("step", 4))
    min_bsz = int(probe.get("min_bsz", step))
    max_bsz = int(probe.get("max_bsz", 192))

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_processor = PromptProcessor(
        tokenizer=tokenizer,
        system_prompt=LLMModel.SYSTEM_PROMPT,
        prompt_format="raw_chunk_first",
    )

    plans = []
    for index, case in enumerate(iter_eval_cases(config)):
        source_result = select_source_result(
            source_results=source_results,
            preprocess_name=preprocess_name,
            case=case,
        )
        source_output = resolve_source_output(
            source_run_dir, stringify(source_result["OUTPUT_FILE"])
        )
        longest_prompt, max_prompt_tokens, mean_prompt_tokens = find_longest_prompt(
            output_path=source_output,
            tokenizer=tokenizer,
            prompt_processor=prompt_processor,
        )
        logged_mean = float(source_result["avg_nocache_model_input_len"])
        if round(mean_prompt_tokens, 4) != round(logged_mean, 4):
            raise ValueError(
                f"case {index} reconstructed mean {mean_prompt_tokens:.4f} "
                f"does not match logged mean {logged_mean:.4f}"
            )
        initial_bsz = choose_initial_bsz(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            target_padded_tokens=target_padded_tokens,
            step=step,
            min_bsz=min_bsz,
            max_bsz=max_bsz,
            override=case.get("PROBE_INITIAL_BSZ", ""),
        )
        plan = CasePlan(
            index=index,
            case=case,
            source_result=source_result,
            source_output=source_output,
            longest_prompt=longest_prompt,
            max_prompt_tokens=max_prompt_tokens,
            mean_prompt_tokens=mean_prompt_tokens,
            initial_bsz=initial_bsz,
        )
        plans.append(plan)
        print(
            f"[plan] case={index} max_prompt_tokens={max_prompt_tokens} "
            f"mean_prompt_tokens={mean_prompt_tokens:.4f} "
            f"initial_bsz={initial_bsz} env={case}",
            flush=True,
        )

    if args.dry_run:
        return 0

    results_root = Path(config.get("results_root", "outputs/grid_search"))
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    run_dir = results_root / stringify(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    probe_results_path = run_dir / "max_bsz_probe.jsonl"
    restored_count = restore_completed_probes(
        probe_results_path,
        plans,
        max_new_tokens=max_new_tokens,
        target_padded_tokens=target_padded_tokens,
        step=step,
        min_bsz=min_bsz,
        max_bsz=max_bsz,
    )

    pending_plans = [plan for plan in plans if plan.selected_bsz is None]
    if pending_plans:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for max-bsz probing")
        if restored_count:
            print(
                f"[probe-resume] completed={restored_count}/{len(plans)} "
                f"remaining={len(pending_plans)}",
                flush=True,
            )
        llm = LLMModel(
            model_name=model_name,
            load_in_4bit=coerce_bool(fixed_env.get("MODEL_LOAD_IN_4BIT", False)),
            prompt_format="raw_chunk_first",
        )
        try:
            for plan in pending_plans:
                candidate = plan.initial_bsz
                attempts = []
                while candidate >= min_bsz:
                    print(
                        f"[probe] case={plan.index} bsz={candidate} "
                        f"prompt_tokens={plan.max_prompt_tokens}",
                        flush=True,
                    )
                    try:
                        memory = run_forced_generation_probe(
                            llm=llm,
                            prompt=plan.longest_prompt,
                            batch_size=candidate,
                            max_new_tokens=max_new_tokens,
                        )
                        attempts.append({"bsz": candidate, "oom": False, **memory})
                        plan.selected_bsz = candidate
                        print(
                            f"[probe-ok] case={plan.index} selected_bsz={candidate} "
                            f"peak_reserved_bytes={memory['peak_reserved_bytes']}",
                            flush=True,
                        )
                        break
                    except BaseException as error:
                        if not is_cuda_oom(error):
                            raise
                        attempts.append(
                            {"bsz": candidate, "oom": True, "error": str(error)}
                        )
                        print(
                            f"[probe-oom] case={plan.index} bsz={candidate}", flush=True
                        )
                        clear_cuda_after_probe()
                        candidate -= step
                if plan.selected_bsz is None:
                    raise RuntimeError(
                        f"case {plan.index} OOMed through minimum batch size {min_bsz}"
                    )
                append_jsonl(
                    probe_results_path,
                    {
                        "case_index": plan.index,
                        "case": plan.case,
                        "source_output": str(plan.source_output),
                        "max_prompt_tokens": plan.max_prompt_tokens,
                        "mean_prompt_tokens": plan.mean_prompt_tokens,
                        "max_new_tokens": max_new_tokens,
                        "target_padded_tokens": target_padded_tokens,
                        "step": step,
                        "min_bsz": min_bsz,
                        "max_bsz": max_bsz,
                        "initial_bsz": plan.initial_bsz,
                        "selected_bsz": plan.selected_bsz,
                        "attempts": attempts,
                    },
                )
        finally:
            del llm
            clear_cuda_after_probe()
    else:
        print(f"[probe-resume] completed={len(plans)}/{len(plans)}", flush=True)

    selected_grid = make_selected_grid(config, plans)
    selected_grid_path = run_dir / "selected_grid.yaml"
    write_or_validate_selected_grid(selected_grid_path, selected_grid)
    if args.probe_only:
        return 0

    replace_process_with_grid_eval(selected_grid_path)


if __name__ == "__main__":
    raise SystemExit(main())
