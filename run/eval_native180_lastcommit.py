"""Run last-commit production evaluation with explicit artifact/model adapters."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def generate(cfg):
    import numpy as np
    import torch
    from transformers import set_seed
    import transformers.models.gemma2.modeling_gemma2 as gemma2

    for name in ["GEMMA2_START_DOCSTRING", "GEMMA2_INPUTS_DOCSTRING"]:
        if not hasattr(gemma2, name):
            setattr(gemma2, name, "")
    torch.set_num_threads(32)
    set_seed(123)
    import model
    from prompt import PromptProcessor
    from compressor import factory
    from compressor.methods.dense import DenseSlidingSubchunkCompressor
    from entrypoint import eval as entrypoint
    from vectordb import ChromaDB

    folder = Path(cfg["output"])
    artifacts = Path(cfg["artifacts"])
    con = sqlite3.connect(
        (artifacts / "regions.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    completed = json.load(open(artifacts / "regions_complete.json"))
    assert completed["tokenizer"] == "lightonai/LateOn" and completed["window"] == 180
    coverage = None
    if (artifacts / "lateon_coverage.npy").exists():
        coverage = set(np.load(artifacts / "lateon_coverage.npy").tolist())

    class NativeMetadataDB(ChromaDB):
        def find_top_k_docs(self, *args, **kwargs):
            batches = super().find_top_k_docs(*args, **kwargs)
            for docs in batches:
                for doc in docs:
                    row = con.execute(
                        "SELECT starts,ends FROM chunks WHERE id=?", (str(doc.id),)
                    ).fetchone()
                    if row is None:
                        raise ValueError(
                            f"Native region artifact does not cover retrieved chunk {doc.id!r}; extend materialization, never drop the chunk"
                        )
                    if cfg["scorer"] == "lateon" and coverage is not None:
                        missing = [
                            c.id for c in doc.cacheables if int(c.id) not in coverage
                        ]
                        if missing:
                            raise ValueError(f"Unencoded LateOn source IDs: {missing}")
                    doc.metadata = {
                        **doc.metadata,
                        "region_starts": json.loads(row[0]),
                        "region_ends": json.loads(row[1]),
                    }
            return batches

    entrypoint.ChromaDB = NativeMetadataDB
    if cfg["scorer"] == "denseon":

        class ArtifactDense(DenseSlidingSubchunkCompressor):
            def __init__(self):
                super().__init__()
                complete = json.load(open(artifacts / "dense_complete.json"))
                vectors = np.load(artifacts / "dense_vectors.npy", mmap_mode="r")
                assert vectors.shape == (complete["rows"], 768)
                for i, rid, text in con.execute(
                    "SELECT rowid,id,text FROM regions ORDER BY rowid"
                ):
                    self._runtime_cacheable_embeddings[rid] = (
                        text,
                        torch.tensor(vectors[i].copy(), dtype=torch.float32),
                    )
                print(
                    "DenseOn prepared region vectors loaded",
                    len(self._runtime_cacheable_embeddings),
                    flush=True,
                )

            def _embed_unique_cacheables(self, cacheables):
                for c in cacheables:
                    if str(c.id) not in self._runtime_cacheable_embeddings:
                        raise ValueError(f"Dense region is not materialized: {c.id!r}")
                return super()._embed_unique_cacheables(cacheables)

        factory.COMPRESSOR_TYPES["dense_sliding_subchunk"] = ArtifactDense
    if cfg["prompt_condition"] == "current_chat":

        class ExplicitContextPrompt(PromptProcessor):
            def _build_chat_prompt(self, query, passages):
                content = f"Context:\n{self.format_passages(passages)}Question:\n{query.strip()}"
                text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if self.tokenizer.bos_token and text.startswith(
                    self.tokenizer.bos_token
                ):
                    text = text[len(self.tokenizer.bos_token) :]
                return text

        model.PromptProcessor = ExplicitContextPrompt
        model.LLMModel.SYSTEM_PROMPT = cfg["system_prompt"]
    else:
        assert model.LLMModel.SYSTEM_PROMPT == cfg["system_prompt"]
    save(
        folder / "execution_settings.json",
        dict(
            base_commit=cfg["base_commit"],
            pipeline="HEAD entrypoint.eval.main; live Chroma retrieval -> production compression -> generation",
            region_tokenizer="lightonai/LateOn",
            window=180,
            prompt_condition=cfg["prompt_condition"],
            system_prompt=cfg["system_prompt"],
            scorer=cfg["scorer"],
            lateon_storage="2bit",
            generation_batch=24,
        ),
    )
    entrypoint.main(
        query_file=cfg["queries"],
        answer_file=cfg["answers"],
        db_dir=cfg["env"]["DB_DIR"],
        cache_dir=str(folder),
        dataset=cfg["dataset"],
        top_k=20,
        use_past_cache=False,
        bsz=24,
        max_new_tokens=20,
        total_num=None,
        output_file=str(folder / "predictions.jsonl"),
        model_name=cfg["env"]["MODEL_NAME"],
        compress_method=cfg["env"]["COMPRESS_METHOD"],
        use_cleaner=False,
        model_load_in_4bit=False,
        prompt_format=cfg["env"]["PROMPT_FORMAT"],
    )


def summarize(cfg):
    from acc_metric import build_evaluator, extract_prediction_from_record

    folder = Path(cfg["output"])
    preds = [json.loads(s) for s in (folder / "predictions.jsonl").open()]
    qs = [json.loads(s) for s in open(cfg["queries"])]
    answers = [json.loads(s) for s in open(cfg["answers"])]
    assert len(preds) == len(qs) == len(answers) == cfg["count"]
    evaluator = build_evaluator(cfg["dataset"])
    scores = []
    for p, q, a in zip(preds, qs, answers, strict=True):
        assert p["id"] == q["id"] == a["id"] and p["question"] == q["query"]
        em, f1 = evaluator.score_prediction(
            extract_prediction_from_record(p, use_cleaner=False), a
        )
        scores.append(dict(id=p["id"], em=em, f1=f1))
    (folder / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    log = (folder / "run.log").read_text()
    match = re.search(r"avg no-cache model input len: ([\d.]+)", log)
    assert match
    result = dict(
        count=len(scores),
        f1=sum(s["f1"] for s in scores) / len(scores),
        exact_match=sum(s["em"] for s in scores) / len(scores),
        avg_prompt_tokens=float(match[1]),
    )
    save(folder / "results.json", result)
    grid = Path(cfg["code"]) / "run/grid_search/eval.py"
    spec = importlib.util.spec_from_file_location("native_grid_report", grid)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reporter = object.__new__(module.GridRunner)
    reporter.summary2_csv = folder / "summary2.csv"
    row = (
        cfg["env"]
        | reporter._extract_eval_summary(log)
        | dict(
            f1=result["f1"],
            exact_match=result["exact_match"],
            avg_valid_input_len=result["avg_prompt_tokens"],
        )
    )
    reporter._write_summary2_csv([row])
    save(folder / "summary.json", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["run", "generate"], default="run")
    args = parser.parse_args()
    cfg = json.load(args.config.open())
    os.environ.update(cfg["env"])
    sys.path.insert(0, str(Path(cfg["code"]) / "src"))
    folder = Path(cfg["output"])
    folder.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        generate(cfg)
    else:
        with (folder / "run.log").open("a") as f:
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
                stdout=f,
                stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONPATH": str(Path(cfg["code"]) / "src")},
            )
        summarize(cfg)
