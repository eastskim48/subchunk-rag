from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("probe_max_bsz_and_eval.py")
SPEC = importlib.util.spec_from_file_location("probe_max_bsz_and_eval", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_choose_initial_bsz_rounds_down_to_step() -> None:
    selected = MODULE.choose_initial_bsz(
        max_prompt_tokens=3388,
        max_new_tokens=20,
        target_padded_tokens=65536,
        step=4,
        min_bsz=4,
        max_bsz=128,
    )

    assert selected == 16


def test_case_match_ignores_probe_and_batch_fields() -> None:
    case = {
        "DATA_SUBDIR": "vanilla-128",
        "TOP_K": "10",
        "EVAL_BSZ": "auto",
        "PROBE_INITIAL_BSZ": "44",
    }
    result = {"DATA_SUBDIR": "vanilla-128", "TOP_K": "10", "EVAL_BSZ": "1"}

    assert MODULE.case_matches_result(case, result)


def test_make_selected_grid_removes_probe_controls() -> None:
    config = {
        "run_name": "test",
        "fixed_env": {"EVAL_BSZ": "auto"},
        "auto_eval_bsz": {"enabled": True},
        "max_prompt_bsz_probe": {"source_run_dir": "source"},
        "eval_axes": {"TOP_K": ["10"]},
    }
    plan = MODULE.CasePlan(
        index=0,
        case={"TOP_K": "10", "PROBE_INITIAL_BSZ": "20"},
        source_result={},
        source_output=Path("source.jsonl"),
        longest_prompt="prompt",
        max_prompt_tokens=100,
        mean_prompt_tokens=90.0,
        initial_bsz=20,
        selected_bsz=16,
    )

    selected = MODULE.make_selected_grid(config, [plan])

    assert "auto_eval_bsz" not in selected
    assert "max_prompt_bsz_probe" not in selected
    assert "eval_axes" not in selected
    assert selected["fixed_env"]["EVAL_BSZ"] == "1"
    assert selected["eval_cases"] == [{"TOP_K": "10", "EVAL_BSZ": "16"}]


def make_probe_plan() -> object:
    return MODULE.CasePlan(
        index=0,
        case={"TOP_K": "10"},
        source_result={},
        source_output=Path("/source.jsonl"),
        longest_prompt="prompt",
        max_prompt_tokens=100,
        mean_prompt_tokens=90.0,
        initial_bsz=20,
    )


def test_restore_completed_probes_restores_valid_prefix(tmp_path) -> None:
    plan = make_probe_plan()
    path = tmp_path / "max_bsz_probe.jsonl"
    MODULE.append_jsonl(
        path,
        {
            "case_index": 0,
            "case": {"TOP_K": "10"},
            "source_output": "/source.jsonl",
            "max_prompt_tokens": 100,
            "mean_prompt_tokens": 90.0,
            "max_new_tokens": 20,
            "target_padded_tokens": 65536,
            "initial_bsz": 20,
            "selected_bsz": 16,
            "attempts": [{"bsz": 16, "oom": False}],
        },
    )

    restored = MODULE.restore_completed_probes(
        path,
        [plan],
        max_new_tokens=20,
        target_padded_tokens=65536,
        step=4,
        min_bsz=4,
        max_bsz=128,
    )

    assert restored == 1
    assert plan.selected_bsz == 16


def test_restore_completed_probes_rejects_changed_case(tmp_path) -> None:
    plan = make_probe_plan()
    path = tmp_path / "max_bsz_probe.jsonl"
    MODULE.append_jsonl(
        path,
        {
            "case_index": 0,
            "case": {"TOP_K": "20"},
            "source_output": "/source.jsonl",
            "max_prompt_tokens": 100,
            "mean_prompt_tokens": 90.0,
            "max_new_tokens": 20,
            "target_padded_tokens": 65536,
            "initial_bsz": 20,
            "selected_bsz": 16,
            "attempts": [{"bsz": 16, "oom": False}],
        },
    )

    with pytest.raises(ValueError, match="field case"):
        MODULE.restore_completed_probes(
            path,
            [plan],
            max_new_tokens=20,
            target_padded_tokens=65536,
            step=4,
            min_bsz=4,
            max_bsz=128,
        )


def test_replace_process_with_grid_eval_releases_probe_process(monkeypatch) -> None:
    selected_grid = Path("outputs/run/selected_grid.yaml")
    captured = {}

    def fake_chdir(path: Path) -> None:
        captured["cwd"] = path

    def fake_execve(executable, command, environment) -> None:
        captured["executable"] = executable
        captured["command"] = command
        captured["environment"] = environment
        raise RuntimeError("execve called")

    monkeypatch.setattr(MODULE.os, "chdir", fake_chdir)
    monkeypatch.setattr(MODULE.os, "execve", fake_execve)

    with pytest.raises(RuntimeError, match="execve called"):
        MODULE.replace_process_with_grid_eval(selected_grid)

    assert captured["cwd"] == MODULE.REPO_ROOT
    assert captured["executable"] == sys.executable
    assert captured["command"] == [
        sys.executable,
        str(MODULE.REPO_ROOT / "run/grid_search/eval.py"),
        str(selected_grid),
    ]
    assert captured["environment"] == os.environ.copy()


def test_run_grids_sequentially_continues_after_failure(monkeypatch) -> None:
    grids = [Path("first.yaml"), Path("second.yaml"), Path("third.yaml")]
    commands = []
    return_codes = iter([0, 7, 0])

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_run(command, *, cwd, env):
        commands.append((command, cwd, env))
        return Completed(next(return_codes))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    status = MODULE.run_grids_sequentially(grids, dry_run=False, probe_only=False)

    assert status == 1
    assert [entry[0][-1] for entry in commands] == [
        "first.yaml",
        "second.yaml",
        "third.yaml",
    ]
    assert all(entry[1] == MODULE.REPO_ROOT for entry in commands)
    assert all(entry[2] == os.environ.copy() for entry in commands)
