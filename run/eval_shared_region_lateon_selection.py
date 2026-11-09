"""Attach existing contributor/DISCo selectors to the current main pipeline."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test"), str(ROOT / "run")]


def contributor_class(cfg):
    import torch
    from compressor.methods.colbert import region as module

    # Reuse the existing nested class verbatim, as the existing contributor
    # regression test does, without running its dataset-specific CLI wrapper.
    path = ROOT / "test/eval_colbert_contributor_budget.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "PrunedRegions"
    )
    namespace = dict(
        torch=torch,
        module=module,
        os=os,
        Path=Path,
        json=json,
        repo=ROOT,
        data=Path(cfg["env"]["DB_DIR"]).parent,
        args=SimpleNamespace(validate_contexts=None),
        audit=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["PrunedRegions"]


def generate(cfg):
    from compressor import factory
    from contributor_region_override import RegionOverride
    from run_hotpot_lighton_suite import generate as main_generate

    if cfg["selection_variant"] == "contributors":
        base = contributor_class(cfg)
    else:
        assert cfg["selection_variant"] == "disco_stop"
        from run_lateon_sentence_region_marginal import SentenceRegionMarginalCompressor

        base = SentenceRegionMarginalCompressor

    class SharedWindowSelection(base):
        def __init__(self):
            super().__init__()
            self.shared_regions = RegionOverride(
                self.encoder, cfg["env"]["DB_DIR"], cfg["window"], rebuild=True
            )
            self.region_token_budget = cfg["window"]

        def _cached_sliding_region_specs(self, doc, cacheables):
            return self.shared_regions.specs(doc, cacheables, self.artifact)

    factory.COMPRESSOR_TYPES[cfg["methods"][cfg["method"]]["COMPRESS_METHOD"]] = (
        SharedWindowSelection
    )
    main_generate(cfg, cfg["method"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["generate", "run"], required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    os.environ.update(cfg["env"])
    from build_full_lighton_artifacts import initialize, sha, save

    for path, expected in cfg["input_hashes"].items():
        assert sha(Path(path)) == expected, path
    initialize()
    if args.action == "generate":
        generate(cfg)
        return
    folder = Path(cfg["output"]) / cfg["method"]
    folder.mkdir(exist_ok=True)
    with (folder / "run.log").open("a") as stream:
        subprocess.run(
            [
                sys.executable,
                "-u",
                __file__,
                "--config",
                str(args.config),
                "--action",
                "generate",
            ],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    from run_hotpot_lighton_suite import summarize
    from overnight_worker import write_summary2

    summarize(cfg, cfg["method"])
    write_summary2(cfg, folder)
    save(
        Path(cfg["output"]) / "complete.json",
        json.loads((folder / "results.json").read_text()),
    )


if __name__ == "__main__":
    main()
