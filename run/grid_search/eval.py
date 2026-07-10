from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CompletedStage:
    returncode: int
    stdout: str
    elapsed_sec: float


class GridRunner:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = self._load_config()
        self.run_name = self.config["run_name"]
        self.dataset_prefix = self.config.get("dataset_prefix")
        if "datasets" in self.config:
            datasets = [self._stringify(value) for value in self.config["datasets"]]
            if len(datasets) != 1:
                raise ValueError(
                    "grid runner now supports exactly one dataset per run; "
                    "prepare caches manually and launch each dataset separately"
                )
            self.dataset = datasets[0]
        else:
            self.dataset = self._stringify(self.config["dataset"])
        self.eval_script = self.repo_root / self.config.get(
            "eval_script", "run/eval.sh"
        )
        self.results_dir = (
            self.repo_root
            / self.config.get("results_root", "outputs/grid_search")
            / self.run_name
        )
        self.fixed_env = {
            k: self._stringify(v) for k, v in self.config.get("fixed_env", {}).items()
        }
        self.eval_fixed_env = {
            k: self._stringify(v)
            for k, v in self.config.get("eval_fixed_env", {}).items()
        }
        self.preprocess_groups = self.config.get("preprocess_groups", [])
        self.eval_axes = self.config.get("eval_axes", {})
        self.eval_cases = self.config.get("eval_cases", [])
        self._validate_eval_config()
        self.results_jsonl = self.results_dir / "results.jsonl"
        self.failures_jsonl = self.results_dir / "failures.jsonl"
        self.bsz_probe_jsonl = self.results_dir / "bsz_probes.jsonl"
        dataset_slug = self.dataset.replace("/", "_")
        self.summary_csv = self.results_dir / f"summary-{dataset_slug}.csv"
        self.summary2_csv = self.summary_csv.with_name(
            f"{self.summary_csv.stem}-summary2.csv"
        )
        self.events_log = self.results_dir / "events.log"
        self.logs_dir = self.results_dir / "logs"
        self.bsz_probe_dir = self.results_dir / "bsz_probe_outputs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.bsz_probe_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open() as f:
            return yaml.safe_load(f)

    def _resolve_dataset_path(self, dataset: str) -> Path:
        if self.dataset_prefix:
            return Path(self.dataset_prefix) / dataset
        candidate = Path("/mnt/nvme1/datasets") / dataset
        if candidate.exists():
            return candidate
        return Path(dataset)

    def _write_manifest(self):
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(self.config_path),
            "dataset": self.dataset,
            "eval_script": str(self.eval_script),
            "fixed_env": self.fixed_env,
            "eval_fixed_env": self.eval_fixed_env,
            "auto_eval_bsz": self.config.get("auto_eval_bsz", {}),
            "preprocess_groups": self.preprocess_groups,
            "eval_axes": self.eval_axes,
            "eval_cases": self.eval_cases,
        }
        (self.results_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

    def _validate_eval_config(self) -> None:
        if self.eval_cases and self.eval_axes:
            raise ValueError("eval_cases and eval_axes are mutually exclusive")
        if self.eval_axes and not isinstance(self.eval_axes, dict):
            raise ValueError("eval_axes must be a mapping")
        if self.eval_cases and not isinstance(self.eval_cases, list):
            raise ValueError("eval_cases must be a list of mappings")
        for idx, case in enumerate(self.eval_cases):
            if not isinstance(case, dict):
                raise ValueError(f"eval_cases[{idx}] must be a mapping")

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on"}
        return bool(value)

    def _parse_int_list(self, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = str(value).split(",")
        values = []
        for raw_value in raw_values:
            text = str(raw_value).strip()
            if not text:
                continue
            number = int(text)
            if number <= 0:
                raise ValueError(
                    f"batch-size candidates must be positive, got {number}"
                )
            values.append(number)
        return sorted(set(values))

    def _auto_bsz_config(self) -> dict[str, Any]:
        config = self.config.get("auto_eval_bsz", {}) or {}
        if not isinstance(config, dict):
            raise ValueError("auto_eval_bsz must be a mapping")
        return config

    def _auto_bsz_enabled(self) -> bool:
        config = self._auto_bsz_config()
        fixed_value = self.fixed_env.get("EVAL_BSZ", "").strip().lower()
        return self._coerce_bool(config.get("enabled", False)) or fixed_value in {
            "auto",
            "max",
            "auto_max",
        }

    def _auto_bsz_candidates(self) -> list[int]:
        config = self._auto_bsz_config()
        candidates = self._parse_int_list(config.get("candidates"))
        if not candidates:
            candidates = self._parse_int_list(
                self.fixed_env.get("AUTO_EVAL_BSZ_CANDIDATES")
            )
        if not candidates:
            min_value = config.get("min", self.fixed_env.get("AUTO_EVAL_BSZ_MIN"))
            max_value = config.get("max", self.fixed_env.get("AUTO_EVAL_BSZ_MAX"))
            if min_value is not None and max_value is not None:
                step = int(
                    config.get(
                        "step", self.fixed_env.get("AUTO_EVAL_BSZ_STEP", 1)
                    )
                )
                if step <= 0:
                    raise ValueError(f"auto_eval_bsz.step must be positive, got {step}")
                start = int(min_value)
                stop = int(max_value)
                if start <= 0 or stop <= 0:
                    raise ValueError(
                        "auto_eval_bsz min/max must be positive, "
                        f"got min={start}, max={stop}"
                    )
                if start > stop:
                    raise ValueError(
                        f"auto_eval_bsz min must be <= max, got {start} > {stop}"
                    )
                candidates = list(range(start, stop + 1, step))
                if candidates[-1] != stop:
                    candidates.append(stop)
        return candidates or [1, 2, 4, 8]

    def _auto_bsz_probe_total_num(self) -> int:
        config = self._auto_bsz_config()
        return int(
            config.get(
                "probe_total_num",
                self.fixed_env.get("AUTO_EVAL_BSZ_PROBE_TOTAL_NUM", 8),
            )
        )

    def _auto_bsz_probe_total_num_for_candidate(self, candidate: int) -> int:
        config = self._auto_bsz_config()
        probe_batches = config.get(
            "probe_batches", self.fixed_env.get("AUTO_EVAL_BSZ_PROBE_BATCHES")
        )
        if probe_batches is None:
            total_num = self._auto_bsz_probe_total_num()
        else:
            batches = int(probe_batches)
            if batches <= 0:
                raise ValueError(
                    f"auto_eval_bsz.probe_batches must be positive, got {batches}"
                )
            total_num = candidate * batches
        probe_total_num_max = config.get(
            "probe_total_num_max",
            self.fixed_env.get("AUTO_EVAL_BSZ_PROBE_TOTAL_NUM_MAX"),
        )
        if probe_total_num_max is not None:
            max_total_num = int(probe_total_num_max)
            if max_total_num <= 0:
                raise ValueError(
                    "auto_eval_bsz.probe_total_num_max must be positive, "
                    f"got {max_total_num}"
                )
            total_num = min(total_num, max_total_num)
        return total_num

    def _auto_bsz_search_mode(self) -> str:
        config = self._auto_bsz_config()
        mode = self._stringify(
            config.get("search", self.fixed_env.get("AUTO_EVAL_BSZ_SEARCH", "linear"))
        )
        mode = mode.strip().lower()
        aliases = {
            "": "linear",
            "sequential": "linear",
            "seq": "linear",
            "bounded_binary": "binary",
            "bisect": "binary",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"linear", "binary"}:
            raise ValueError(
                "auto_eval_bsz.search must be one of {'linear', 'binary'}, "
                f"got {mode!r}"
            )
        return mode

    def _is_oom_failure(self, stdout: str) -> bool:
        lowered = stdout.lower()
        patterns = (
            "cuda out of memory",
            "torch.outofmemoryerror",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "hip out of memory",
            "out of memory",
        )
        return any(pattern in lowered for pattern in patterns)

    def _run_bsz_probe(
        self,
        dataset: str,
        merged_eval_env: dict[str, str],
        log_prefix: str,
        candidate: int,
    ) -> CompletedStage:
        probe_total_num = self._auto_bsz_probe_total_num_for_candidate(candidate)
        probe_env = dict(merged_eval_env)
        probe_env["EVAL_BSZ"] = str(candidate)
        probe_env["TOTAL_NUM"] = str(probe_total_num)
        probe_env["OUTPUT_FILE"] = str(
            self.bsz_probe_dir / f"{log_prefix}__bsz={candidate}.jsonl"
        )
        probe_log_prefix = f"{log_prefix}__bsz_probe__bsz={candidate}"
        self._print_status(
            f"[bsz-probe] dataset={dataset} bsz={candidate} total_num={probe_total_num}"
        )
        probe_stage = self._run_script(
            self.eval_script, dataset, probe_env, probe_log_prefix
        )
        probe_payload = {
            "dataset": dataset,
            "log_file": str(self.logs_dir / f"{probe_log_prefix}.log"),
            "returncode": probe_stage.returncode,
            "elapsed_sec": probe_stage.elapsed_sec,
            "EVAL_BSZ": str(candidate),
            "probe_total_num": probe_total_num,
            "search": self._auto_bsz_search_mode(),
            "oom": self._is_oom_failure(probe_stage.stdout),
        }
        probe_payload.update(
            {k: v for k, v in merged_eval_env.items() if k != "EVAL_BSZ"}
        )
        self._append_jsonl(self.bsz_probe_jsonl, probe_payload)
        return probe_stage

    def _select_auto_bsz_linear(
        self,
        dataset: str,
        merged_eval_env: dict[str, str],
        log_prefix: str,
        candidates: list[int],
    ) -> int | None:
        last_ok = None
        for candidate in candidates:
            probe_stage = self._run_bsz_probe(
                dataset, merged_eval_env, log_prefix, candidate
            )
            if probe_stage.returncode == 0:
                last_ok = candidate
                continue
            if self._is_oom_failure(probe_stage.stdout):
                self._print_status(f"[bsz-probe-oom] dataset={dataset} bsz={candidate}")
                break
            self._print_status(
                f"[bsz-probe-failed] dataset={dataset} bsz={candidate} "
                f'rc={probe_stage.returncode} log={self.logs_dir / f"{log_prefix}__bsz_probe__bsz={candidate}.log"}'
            )
            break
        return last_ok

    def _select_auto_bsz_binary(
        self,
        dataset: str,
        merged_eval_env: dict[str, str],
        log_prefix: str,
        candidates: list[int],
    ) -> int | None:
        last_ok = None
        low = 0
        high = len(candidates) - 1
        while low <= high:
            mid = (low + high) // 2
            candidate = candidates[mid]
            probe_stage = self._run_bsz_probe(
                dataset, merged_eval_env, log_prefix, candidate
            )
            if probe_stage.returncode == 0:
                last_ok = candidate
                low = mid + 1
                continue
            if self._is_oom_failure(probe_stage.stdout):
                self._print_status(f"[bsz-probe-oom] dataset={dataset} bsz={candidate}")
                high = mid - 1
                continue
            self._print_status(
                f"[bsz-probe-failed] dataset={dataset} bsz={candidate} "
                f'rc={probe_stage.returncode} log={self.logs_dir / f"{log_prefix}__bsz_probe__bsz={candidate}.log"}'
            )
            break
        return last_ok

    def _select_auto_bsz(
        self, dataset: str, merged_eval_env: dict[str, str], log_prefix: str
    ) -> int | None:
        if not self._auto_bsz_enabled():
            return None

        candidates = self._auto_bsz_candidates()
        search_mode = self._auto_bsz_search_mode()
        self._print_status(
            f"[bsz-search] dataset={dataset} mode={search_mode} candidates={candidates}"
        )
        if search_mode == "binary":
            last_ok = self._select_auto_bsz_binary(
                dataset, merged_eval_env, log_prefix, candidates
            )
        else:
            last_ok = self._select_auto_bsz_linear(
                dataset, merged_eval_env, log_prefix, candidates
            )

        if last_ok is None:
            raise RuntimeError(
                f"auto batch-size probe failed for every candidate {candidates}; "
                f"check {self.bsz_probe_jsonl}"
            )
        self._print_status(f"[bsz-selected] dataset={dataset} EVAL_BSZ={last_ok}")
        self._log_event(
            f"auto_bsz selected dataset={dataset} env={merged_eval_env} EVAL_BSZ={last_ok}"
        )
        return last_ok

    def _log_event(self, message: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.events_log.open("a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _print_status(self, message: str):
        print(message, flush=True)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]):
        with path.open("a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _run_script(
        self,
        script_path: Path,
        dataset: str,
        env_updates: dict[str, str],
        log_prefix: str,
    ) -> CompletedStage:
        env = os.environ.copy()
        env["DATASET"] = dataset
        if self.dataset_prefix:
            env["DATASET_PREFIX"] = self.dataset_prefix
        env.update(env_updates)

        start = time.perf_counter()
        proc = subprocess.run(
            ["bash", str(script_path)],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout or ""
        (self.logs_dir / f"{log_prefix}.log").write_text(stdout)
        return CompletedStage(
            returncode=proc.returncode, stdout=stdout, elapsed_sec=elapsed
        )

    def _iter_eval_envs(self):
        if self.eval_cases:
            for case in self.eval_cases:
                yield {key: self._stringify(value) for key, value in case.items()}
            return

        axes = list(self.eval_axes.keys())
        values = [
            (
                self.eval_axes[key]
                if isinstance(self.eval_axes[key], list)
                else [self.eval_axes[key]]
            )
            for key in axes
        ]
        for combo in itertools.product(*values):
            env = {key: self._stringify(value) for key, value in zip(axes, combo)}
            yield env

    def _extract_eval_summary(self, stdout: str) -> dict[str, Any]:
        lines = stdout.splitlines()
        summary = {}
        decoder = json.JSONDecoder()
        for idx, line in enumerate(lines):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            block = "\n".join(lines[idx:])
            try:
                obj, _ = decoder.raw_decode(block)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "f1" in obj and "exact_match" in obj:
                summary.update(
                    {
                        "count": obj.get("count"),
                        "exact_match": obj.get("exact_match"),
                        "f1": obj.get("f1"),
                        "evaluator": obj.get("evaluator"),
                        "dataset": obj.get("dataset"),
                    }
                )
                break
            if isinstance(obj, dict) and "vanilla" in obj and "compressed" in obj:
                vanilla = obj.get("vanilla") or {}
                compressed = obj.get("compressed") or {}
                delta = obj.get("delta") or {}
                gold = obj.get("gold") or {}
                context_summary = {
                    "count": obj.get("count"),
                    "dataset": obj.get("dataset"),
                    "gold_chunk_counts": gold.get("chunk_counts"),
                    "gold_subchunk_counts": gold.get("subchunk_counts"),
                    "gold_subchunk_rouge_l_recall": gold.get("subchunk_rouge_l_recall"),
                    "vanilla_rouge_l_recall": vanilla.get("rouge_l_recall"),
                    "vanilla_rouge_l_precision": vanilla.get("rouge_l_precision"),
                    "vanilla_selected_tokens": vanilla.get("selected_tokens"),
                    "vanilla_chunk_level_recall": vanilla.get("chunk_level_recall"),
                    "vanilla_chunk_level_precision": vanilla.get(
                        "chunk_level_precision"
                    ),
                    "vanilla_subchunk_level_recall": vanilla.get(
                        "subchunk_level_recall"
                    ),
                    "vanilla_subchunk_level_precision": vanilla.get(
                        "subchunk_level_precision"
                    ),
                    "compressed_rouge_l_recall": compressed.get("rouge_l_recall"),
                    "compressed_rouge_l_precision": compressed.get("rouge_l_precision"),
                    "compressed_token_counts": compressed.get("token_counts"),
                    "compressed_subchunk_counts": compressed.get("subchunk_counts"),
                    "compressed_subchunk_level_recall": compressed.get(
                        "subchunk_level_recall"
                    ),
                    "compressed_subchunk_level_precision": compressed.get(
                        "subchunk_level_precision"
                    ),
                    "recall_drop": delta.get("recall_drop"),
                    "precision_gain": delta.get("precision_gain"),
                    "token_reduction": delta.get("token_reduction"),
                    "context_setup_time_sec": obj.get("setup_time_sec"),
                    "context_run_time_sec": obj.get("run_time_sec"),
                }
                if gold:
                    context_summary.update(
                        {
                            "gold_tokens": gold.get("tokens"),
                        }
                    )
                summary.update(context_summary)
                break
        metrics = {
            "throughput_requests_per_sec": "throughput | requests/sec",
            "throughput_batches_per_sec": "throughput | batches/sec",
            "run_time_sec": "run time",
            "process_total_time_sec": "process total time",
            "score_time_sec": "score time",
            "end_to_end_time_sec": "end-to-end time",
            "retrieval_per_batch_avg_sec": "retrieval time per batch | total",
            "retrieval_query_per_batch_avg_sec": "retrieval query time per batch | total",
            "retrieval_postprocess_per_batch_avg_sec": "retrieval postprocess time per batch | total",
            "retrieval_cacheable_deserialize_per_batch_avg_sec": "retrieval cacheable deserialize time per batch | total",
            "compress_per_batch_avg_sec": "compress time per batch | total",
            "prompt_build_per_batch_avg_sec": "prompt build time per batch | total",
            "prompt_stats_per_batch_avg_sec": "prompt stats time per batch | total",
            "generate_extra_per_batch_avg_sec": "generate extra time per batch | total",
            "prefill_per_batch_avg_sec": "prefill per batch | total",
            "decode_per_batch_avg_sec": "decode per batch | total",
            "other_time_per_batch_avg_sec": "other time per batch | total",
            "time_per_batch_avg_sec": "time per batch| total",
            "avg_valid_cache_len": "avg valid cache len",
            "avg_valid_prefill_input_len": "avg valid prefill input len",
            "avg_cached_rate": "avg cached rate",
            "avg_nocache_model_input_len": "avg no-cache model input len",
            "avg_nocache_query_only_len": "avg no-cache query-only len",
        }
        for key, label in metrics.items():
            for line in lines:
                if not line.startswith(label):
                    continue
                if "avg:" in line:
                    summary[key] = float(line.split("avg:")[1].strip().rstrip("s"))
                else:
                    value = line.split(":", 1)[1].strip().split()[0]
                    try:
                        summary[key] = float(value)
                    except ValueError:
                        pass
                break
        if "avg_valid_prefill_input_len" in summary:
            summary["avg_valid_input_len"] = summary["avg_valid_prefill_input_len"]
        elif "avg_nocache_model_input_len" in summary:
            summary["avg_valid_input_len"] = summary["avg_nocache_model_input_len"]
        self._add_derived_ttft(summary)
        return summary

    def _add_derived_ttft(self, row: dict[str, Any]) -> None:
        required_keys = (
            "time_per_batch_avg_sec",
            "decode_per_batch_avg_sec",
            "generate_extra_per_batch_avg_sec",
        )
        if "ttft_per_batch_avg_sec" in row:
            return
        if any(key not in row for key in required_keys):
            return
        try:
            row["ttft_per_batch_avg_sec"] = (
                float(row["time_per_batch_avg_sec"])
                - float(row["decode_per_batch_avg_sec"])
                - float(row["generate_extra_per_batch_avg_sec"])
            )
        except (TypeError, ValueError):
            return

    def _backfill_summary_row_from_log(self, row: dict[str, Any]) -> dict[str, Any]:
        self._add_derived_ttft(row)
        missing_keys = [
            key
            for key in (
                "throughput_requests_per_sec",
                "throughput_batches_per_sec",
                "ttft_per_batch_avg_sec",
            )
            if key not in row
        ]
        if not missing_keys:
            return row

        log_file = row.get("eval_log_file")
        if not log_file:
            return row
        log_path = Path(log_file)
        if not log_path.exists():
            return row

        log_summary = self._extract_eval_summary(log_path.read_text(errors="replace"))
        for key in missing_keys:
            if key in log_summary:
                row[key] = log_summary[key]
        self._add_derived_ttft(row)
        return row

    @staticmethod
    def _normalize_summary_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        if normalized.get("COMPRESS_METHOD") in {None, ""} and normalized.get(
            "DATA_SUBDIR"
        ):
            normalized["COMPRESS_METHOD"] = normalized["DATA_SUBDIR"]
        return normalized

    @staticmethod
    def _summary2_method_name(method: Any, data_subdir: Any = None) -> Any:
        if method is None or method == "":
            return data_subdir
        if method in {"compare_all_materialized", "compare_all"}:
            return "embed_sim"
        return method

    def _to_summary2_row(self, row: dict[str, Any]) -> dict[str, Any]:
        input_len = row.get("avg_valid_input_len")
        if input_len is None:
            input_len = row.get("avg_nocache_model_input_len")
        return {
            "COMPRESS_METHOD": self._summary2_method_name(
                row.get("COMPRESS_METHOD"), row.get("DATA_SUBDIR")
            ),
            "GLOBAL_TOP_R": row.get("GLOBAL_TOP_R"),
            "TOP_K": row.get("TOP_K"),
            "COLBERT_FINAL_TOKEN_BUDGET": row.get("COLBERT_FINAL_TOKEN_BUDGET"),
            "RETAIN_TOKEN_RATIO": row.get("RETAIN_TOKEN_RATIO"),
            "input_len": input_len,
            "exact_match": row.get("exact_match"),
            "f1": row.get("f1"),
            "prefill_per_batch_avg_sec": row.get("prefill_per_batch_avg_sec"),
            "decode_per_batch_avg_sec": row.get("decode_per_batch_avg_sec"),
            "elapsed_sec": row.get("elapsed_sec"),
            "end_to_end_time_sec": row.get("end_to_end_time_sec"),
            "time_per_batch_avg_sec": row.get("time_per_batch_avg_sec"),
            "throughput_requests_per_sec": row.get("throughput_requests_per_sec"),
            "throughput_batches_per_sec": row.get("throughput_batches_per_sec"),
            "compress_time_per_batch": row.get("compress_per_batch_avg_sec"),
            "TTFT_per_batch": row.get("ttft_per_batch_avg_sec"),
            "Retrieval_per_batch": row.get("retrieval_per_batch_avg_sec"),
        }

    def _write_summary2_csv(self, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "COMPRESS_METHOD",
            "GLOBAL_TOP_R",
            "TOP_K",
            "COLBERT_FINAL_TOKEN_BUDGET",
            "RETAIN_TOKEN_RATIO",
            "input_len",
            "exact_match",
            "f1",
            "prefill_per_batch_avg_sec",
            "decode_per_batch_avg_sec",
            "elapsed_sec",
            "end_to_end_time_sec",
            "time_per_batch_avg_sec",
            "throughput_requests_per_sec",
            "throughput_batches_per_sec",
            "compress_time_per_batch",
            "TTFT_per_batch",
            "Retrieval_per_batch",
        ]
        with self.summary2_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._to_summary2_row(row) for row in rows)

    def _write_summary_csv(self):
        rows = []
        if not self.results_jsonl.exists():
            return
        with self.results_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(
                    self._normalize_summary_row_for_csv(
                        self._backfill_summary_row_from_log(json.loads(line))
                    )
                )
        if not rows:
            return
        keys = sorted({key for row in rows for key in row.keys()})
        with self.summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        self._write_summary2_csv(rows)

    def run(self):
        dataset = self.dataset
        dataset_slug = dataset.replace("/", "_")
        for preprocess_group in self.preprocess_groups:
            preprocess_name = preprocess_group["name"]
            preprocess_env = {
                k: self._stringify(v)
                for k, v in preprocess_group.get("env", {}).items()
            }
            merged_group_env = dict(self.fixed_env)
            merged_group_env.update(preprocess_env)

            for eval_env in self._iter_eval_envs():
                merged_eval_env = dict(merged_group_env)
                merged_eval_env.update(self.eval_fixed_env)
                merged_eval_env.update(eval_env)
                eval_name = "__".join(f"{k.lower()}={v}" for k, v in eval_env.items())
                log_prefix_base = (
                    f"{dataset_slug}__eval__{preprocess_name}__{eval_name}".replace(
                        "/", "_"
                    )
                )
                auto_bsz = self._select_auto_bsz(
                    dataset, merged_eval_env, log_prefix_base
                )
                if auto_bsz is not None:
                    merged_eval_env["EVAL_BSZ"] = str(auto_bsz)
                    candidates = [
                        candidate
                        for candidate in self._auto_bsz_candidates()
                        if candidate <= auto_bsz
                    ]
                else:
                    candidates = []
                log_prefix = (
                    f"{dataset_slug}__eval__{preprocess_name}__{eval_name}".replace(
                        "/", "_"
                    )
                )
                self._print_status(
                    f"[start] dataset={dataset} eval={preprocess_name} "
                    f'env={eval_env} EVAL_BSZ={merged_eval_env.get("EVAL_BSZ", "")}'
                )
                self._log_event(
                    f"start dataset={dataset} eval={preprocess_name} env={merged_eval_env}"
                )
                run_attempts = []
                while True:
                    attempt_log_prefix = log_prefix
                    if auto_bsz is not None:
                        attempt_log_prefix = (
                            f'{log_prefix}__bsz={merged_eval_env["EVAL_BSZ"]}'
                        )
                    eval_stage = self._run_script(
                        self.eval_script, dataset, merged_eval_env, attempt_log_prefix
                    )
                    run_attempts.append(
                        {
                            "EVAL_BSZ": merged_eval_env.get("EVAL_BSZ"),
                            "returncode": eval_stage.returncode,
                            "elapsed_sec": eval_stage.elapsed_sec,
                            "log_file": str(
                                self.logs_dir / f"{attempt_log_prefix}.log"
                            ),
                            "oom": self._is_oom_failure(eval_stage.stdout),
                        }
                    )
                    if eval_stage.returncode == 0:
                        log_prefix = attempt_log_prefix
                        break
                    if auto_bsz is None or not self._is_oom_failure(eval_stage.stdout):
                        log_prefix = attempt_log_prefix
                        break
                    current_bsz = int(merged_eval_env["EVAL_BSZ"])
                    lower_candidates = [
                        candidate for candidate in candidates if candidate < current_bsz
                    ]
                    if not lower_candidates:
                        log_prefix = attempt_log_prefix
                        break
                    retry_bsz = lower_candidates[-1]
                    merged_eval_env["EVAL_BSZ"] = str(retry_bsz)
                    self._print_status(
                        f"[oom-retry] dataset={dataset} eval={preprocess_name} "
                        f"env={eval_env} retry_EVAL_BSZ={retry_bsz}"
                    )
                    self._log_event(
                        f"oom_retry dataset={dataset} eval={preprocess_name} "
                        f"env={merged_eval_env} retry_EVAL_BSZ={retry_bsz}"
                    )
                if eval_stage.returncode != 0:
                    payload = {
                        "stage": "eval",
                        "status": "failed",
                        "dataset": dataset,
                        "preprocess_name": preprocess_name,
                        "eval_env": merged_eval_env,
                        "returncode": eval_stage.returncode,
                        "elapsed_sec": eval_stage.elapsed_sec,
                        "log_file": str(self.logs_dir / f"{log_prefix}.log"),
                        "attempts": run_attempts,
                    }
                    self._append_jsonl(self.failures_jsonl, payload)
                    self._log_event(
                        f"failed dataset={dataset} eval={preprocess_name} env={merged_eval_env}"
                    )
                    self._print_status(
                        f"[failed] dataset={dataset} eval={preprocess_name} "
                        f"env={eval_env} rc={eval_stage.returncode} "
                        f'log={self.logs_dir / f"{log_prefix}.log"}'
                    )
                    continue

                summary = self._extract_eval_summary(eval_stage.stdout)
                result = {
                    "run_name": self.run_name,
                    "dataset": dataset,
                    "preprocess_name": preprocess_name,
                    "status": "ok",
                    "eval_log_file": str(self.logs_dir / f"{log_prefix}.log"),
                    "elapsed_sec": eval_stage.elapsed_sec,
                    "EVAL_BSZ": merged_eval_env.get("EVAL_BSZ"),
                }
                result.update(preprocess_env)
                result.update(eval_env)
                result.update(summary)
                self._append_jsonl(self.results_jsonl, result)
                self._write_summary_csv()
                self._log_event(
                    f"completed dataset={dataset} eval={preprocess_name} env={merged_eval_env}"
                )
                metric_bits = []
                if "f1" in summary:
                    metric_bits.append(f"f1={summary['f1']}")
                if "exact_match" in summary:
                    metric_bits.append(f"em={summary['exact_match']}")
                if "compressed_rouge_l_recall" in summary:
                    metric_bits.append(
                        f"crecall={summary['compressed_rouge_l_recall']}"
                    )
                if "recall_drop" in summary:
                    metric_bits.append(f"drop={summary['recall_drop']}")
                if "compressed_rouge_l_precision" in summary:
                    metric_bits.append(
                        f"cprec={summary['compressed_rouge_l_precision']}"
                    )
                metric_text = " ".join(metric_bits)
                if metric_text:
                    metric_text = f" {metric_text}"
                self._print_status(
                    f"[done] dataset={dataset} eval={preprocess_name} "
                    f"env={eval_env}{metric_text} "
                    f"elapsed={eval_stage.elapsed_sec:.1f}s"
                )


def main(config: str = "run/grid_search/grid.yaml"):
    GridRunner(config).run()


if __name__ == "__main__":
    main(*sys.argv[1:])
