"""Adapters to existing experiment runners; no changes to scoring or prompts."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test")]


def write_summary2(cfg, folder):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "overnight_grid_report", ROOT / "run/grid_search/eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reporter = object.__new__(module.GridRunner)
    row = cfg["env"] | cfg.get("methods", {}).get(cfg["method"], {})
    row.update(reporter._extract_eval_summary((folder / "run.log").read_text()))
    result = json.loads((folder / "results.json").read_text())
    if "overall" in result:
        row.update(f1=result["overall"]["f1"], exact_match=result["overall"]["em"])
    else:
        row.update(f1=result["f1"], exact_match=result["exact_match"])
    row["avg_valid_input_len"] = result.get(
        "avg_prompt_tokens", result.get("mean_prompt_tokens")
    )
    row["RETAIN_TOKEN_RATIO"] = cfg.get("ratio", row.get("RETAIN_TOKEN_RATIO"))
    row["TOP_K"] = int(cfg["env"].get("TOP_K", "20"))
    row["COMPRESS_METHOD"] = row.get("COMPRESS_METHOD", cfg["method"])
    reporter.summary2_csv = folder / "summary2.csv"
    reporter._write_summary2_csv([row])
    (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=[
            "run",
            "covid_run",
            "generate",
            "summarize",
            "covid_select",
            "check_regions",
        ],
        required=True,
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if "jina" in json.dumps(cfg).lower():
        raise ValueError("Retired model/tokenizer dependency in experiment config")
    if "CONTRIBUTOR_WINDOW_TOKENIZER" in cfg["env"]:
        raise ValueError("Region tokenizer overrides are no longer supported")
    os.environ.update(cfg["env"])
    from build_full_lighton_artifacts import initialize, sha, save

    for name, expected in cfg.get("input_hashes", {}).items():
        assert sha(Path(name)) == expected, name
    if args.action in {"run", "covid_run"}:
        folder = Path(cfg["output"]) / cfg["method"]
        folder.mkdir(parents=True, exist_ok=True)
        if args.action == "run":
            commands = [
                (
                    "run.log",
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--config",
                        str(args.config),
                        "--action",
                        "generate",
                    ],
                ),
                (
                    "summarize.log",
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--config",
                        str(args.config),
                        "--action",
                        "summarize",
                    ],
                ),
            ]
        else:
            commands = [
                (
                    "selection.log",
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--config",
                        str(args.config),
                        "--action",
                        "covid_select",
                    ],
                ),
                (
                    "run.log",
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "test/run_covid_qa_evaluation.py"),
                        "--questions",
                        str(Path(cfg["root"]) / "test.jsonl"),
                        "--contexts",
                        str(folder / "contexts.jsonl"),
                        "--output",
                        str(folder),
                        "--batch-size",
                        "20",
                        "--stage",
                        "generate",
                        "--protocol-description",
                        cfg["notes"],
                    ],
                ),
            ]
        for log_name, command in commands:
            with (folder / log_name).open("a") as stream:
                subprocess.run(
                    command, check=True, stdout=stream, stderr=subprocess.STDOUT
                )
        write_summary2(cfg, folder)
        return
    if args.action == "covid_select":
        initialize()
        from run_covid_all_chunks import select

        select(
            Path(cfg["root"]),
            Path(cfg["output"]),
            cfg["method"],
            cfg["ratio"],
            Path(cfg["snapshot"]),
        )
        return
    method = cfg["method"]
    folder = Path(cfg["output"]) / method
    if args.action == "summarize":
        if cfg["dataset"] == "bioasq":
            result = json.loads((folder / "results.json").read_text())
            match = re.search(
                r"avg no-cache model input len: ([\d.]+)",
                (folder / "run.log").read_text(),
            )
            assert match, "Missing mean prompt length"
            result["avg_prompt_tokens"] = float(match[1])
            save(folder / "results.json", result)
        else:
            from run_hotpot_lighton_suite import summarize

            summarize(cfg, method)
        return
    initialize()
    if cfg.get("snapshot"):
        from entrypoint import eval as entrypoint
        from run_nq_materialized_dense_regions import SnapshotVectorDB

        entrypoint.ChromaDB = lambda **kw: SnapshotVectorDB(Path(cfg["snapshot"]))
    if cfg.get("window"):
        from compressor import factory
        from compressor.methods.colbert.region import ColBERTSlidingRegionCompressor
        from contributor_region_override import RegionOverride

        class WindowSweep(ColBERTSlidingRegionCompressor):
            def __init__(self):
                super().__init__()
                self.override = RegionOverride(
                    self.encoder,
                    cfg["env"]["DB_DIR"],
                    cfg["window"],
                    rebuild=cfg.get("region_membership") == "encoder_rebuild",
                )
                self.region_token_budget = cfg["window"]
                save(
                    Path(cfg["output"]) / "region_configuration.json",
                    dict(
                        window=self.region_token_budget,
                        region_tokenizer=self.encoder.model_name,
                        token_overhead=self.encoder.doc_token_overhead,
                        rebuilt_from_encoder=self.override.rebuild,
                    ),
                )

            def _cached_sliding_region_specs(self, doc, cacheables):
                specs = self.override.specs(doc, cacheables, self.artifact)
                audit = Path(cfg["output"]) / "first_chunk_regions.json"
                if not audit.exists():
                    counts = self.encoder.token_counts_without_specials(
                        [c.text for c in cacheables]
                    )
                    save(
                        audit,
                        dict(
                            chunk_id=str(doc.id),
                            region_tokenizer=self.encoder.model_name,
                            window=cfg["window"],
                            regions=[
                                dict(
                                    indices=list(indices),
                                    sentence_count=len(indices),
                                    content_tokens=sum(counts[i] for i in indices),
                                )
                                for _, indices in specs
                            ],
                        ),
                    )
                return specs

        factory.COMPRESSOR_TYPES["colbert_sliding_region"] = WindowSweep
    if args.action == "check_regions":
        from contributor_region_override import RegionOverride
        from colbert_artifact import ColBERTWindowArtifact
        from materialize.db_cacheables import iter_db_records
        from subchunk_store import cacheables_from_metadata
        from chunk import RetrievableChunk

        from encoder.colbert import ColBERTEncoder, default_colbert_repo_path

        encoder = ColBERTEncoder(
            model_name=cfg["env"]["COLBERT_MODEL_NAME"],
            device="cpu",
            batch_size=1,
            repo_path=default_colbert_repo_path(),
        )
        rebuild = cfg.get("region_membership") == "encoder_rebuild"
        override = RegionOverride(
            encoder, cfg["env"]["DB_DIR"], cfg["window"], rebuild=rebuild
        )
        artifact = ColBERTWindowArtifact(cfg["env"]["COLBERT_WINDOW_DIR"])
        checked = 0
        for cid, metadata in iter_db_records(cfg["env"]["DB_DIR"]):
            cacheables = cacheables_from_metadata(metadata, override.subchunks)
            doc = RetrievableChunk(
                id=cid, text="", metadata=metadata, cacheables=cacheables
            )
            specs = override.specs(doc, cacheables, artifact)
            assert all(indices for _, indices in specs)
            checked += 1
            if checked == 100:
                break
        save(
            Path(cfg["output"]) / "region_preflight.json",
            dict(
                checked=checked,
                stored_baseline_checked=not rebuild,
                region_membership="encoder_rebuild" if rebuild else "stored_baseline",
                region_tokenizer=encoder.model_name,
                window=cfg["window"],
            ),
        )
        return
    if cfg["dataset"] == "bioasq":
        from run_bioasq_fixed180 import worker

        worker(cfg, method)
    else:
        from run_hotpot_lighton_suite import generate

        generate(cfg, method)


if __name__ == "__main__":
    main()
