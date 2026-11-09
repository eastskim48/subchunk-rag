"""CPU-only validation of the twelve prepared native grid configurations."""

import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from compressor.factory import COMPRESSOR_TYPES
from subchunk_store import unpack_regions
from utils import parse_json_query_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    counts = {"nq": 2390, "newsqa": 4293, "bioasq": 4366, "covid": 2019}
    yamls = sorted(Path(__file__).parent.glob("[0-9][0-9]_*.yaml"))
    assert len(yamls) == 12
    spec = importlib.util.spec_from_file_location(
        "native_grid_validation", ROOT / "run/grid_search/eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = []
    for position, path in enumerate(yamls):
        cfg = yaml.safe_load(path.read_text())
        env = cfg["fixed_env"]
        dataset = cfg["run_name"]
        method = ["reranker", "reranker_cass", ""][position // 4]
        assert dataset == ["nq", "newsqa", "bioasq", "covid"][position % 4]
        assert cfg["eval_script"] == "run/eval.sh"
        assert env["COMPRESS_METHOD"] == method
        assert not method or method in COMPRESSOR_TYPES
        assert env["EVAL_BSZ"] == "1" and env["PROMPT_FORMAT"] == "chat_system_user"
        assert env["EVAL_USE_PAST_CACHE"] == "False"
        assert env["FINAL_TOKEN_BUDGET"] == (
            "1000" if method == "reranker_cass" else ""
        )
        assert env["TOP_K"] == ("2" if not method else "100")
        assert not env["RETAIN_TOKEN_RATIO"] and not env["RETRIEVAL_SNAPSHOT"]
        runner = module.GridRunner.__new__(module.GridRunner)
        runner.eval_cases, runner.eval_axes = cfg["eval_cases"], {}
        runner._validate_eval_config()
        with Path(env["QUERY_FILE"]).open() as handle:
            questions = [
                parse_json_query_record(line) for line in handle if line.strip()
            ]
        with Path(env["ANSWER_FILE"]).open() as handle:
            answers = [json.loads(line) for line in handle if line.strip()]
        assert len(questions) == len(answers) == counts[dataset]
        assert [str(q["id"]) for q in questions] == [str(a["id"]) for a in answers]
        db = Path(env["DB_DIR"])
        manifest = json.loads((db / "build_manifest.json").read_text())
        assert manifest["retrievable_chunk_size"] == 512
        assert manifest["embedding_backend"] == "bge_small_v1_5"
        assert manifest["splitter"] == (
            "sentence" if method == "reranker_cass" else "fixed_size"
        )
        assert (db / "subchunks.json").is_file()
        with sqlite3.connect(
            (db / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True
        ) as con:
            entries = con.execute("select count(*) from embeddings").fetchone()[0]
            assert entries >= int(env["TOP_K"])
            row = con.execute("select id from embeddings limit 1").fetchone()[0]
            metadata = {
                k: (i if i is not None else s)
                for k, i, s in con.execute(
                    "select key,int_value,string_value from embedding_metadata where id=?",
                    (row,),
                )
            }
            assert type(metadata["subchunk_start"]) is int
            assert metadata["subchunk_end"] > metadata["subchunk_start"]
            if method == "reranker_cass":
                for key in ["region_starts", "region_ends"]:
                    metadata[key] = [
                        r[0]
                        for r in con.execute(
                            "select int_value from embedding_metadata_array where id=? and key=? order by rowid",
                            (row, key),
                        )
                    ]
                start, end = metadata["subchunk_start"], metadata["subchunk_end"]
                assert unpack_regions(metadata, list(range(start, end)))
        if method:
            model = Path(env["RERANKER_MODEL_NAME"])
            assert (model / "model.safetensors").exists()
            config = json.loads((model / "config.json").read_text())
            assert config["architectures"] == ["JinaForRanking"]
            assert env["RERANKER_TOP_N"] == ("20" if method == "reranker_cass" else "2")
            assert env["RERANKER_ATTN_IMPLEMENTATION"] == "flash_attention_2"
            import flash_attn
        if method == "reranker_cass":
            artifact = Path(env["COLBERT_WINDOW_DIR"])
            index = json.loads((artifact / "index.json").read_text())
            assert index["window_token_budget"] == 180
            assert index["model_name"] == "lightonai/LateOn"
            assert (artifact / index["db_manifest"]["path"]).resolve() == (
                db / "build_manifest.json"
            ).resolve()
            data_index = json.loads(
                (artifact / index["data_dir"] / "index.json").read_text()
            )
            assert "2bit" in data_index["format"]
        records.append(
            dict(
                yaml=str(path.relative_to(ROOT)),
                method=method or "vanilla",
                dataset=dataset,
                queries=len(questions),
                db_entries=entries,
                db=str(db),
                valid=True,
            )
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(dict(gpu_inference_run=False, conditions=records), indent=2) + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
