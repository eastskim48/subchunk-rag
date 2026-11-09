"""Verify this requested five-case grid without running answer generation."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/temp/vanilla_db_comparison_20261105"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preparation = json.loads((OUT / "preparation.json").read_text())
    for name, expected in preparation["source_sha256"].items():
        assert digest(REPO / name) == expected, name

    checked = []
    for case in preparation["cases"]:
        config = yaml.safe_load(Path(case["config"]).read_text())
        env = config["fixed_env"]
        for name, expected in {
            "TOP_K": "20",
            "EVAL_BSZ": "1",
            "COMPRESS_METHOD": "",
            "FINAL_TOKEN_BUDGET": "",
            "RETAIN_TOKEN_RATIO": "",
            "RETRIEVAL_SNAPSHOT": "",
            "PARENT_MAPPING_FILE": "",
            "RERANKER_MODEL_NAME": "",
            "EVAL_USE_PAST_CACHE": "False",
            "USE_CLEANER": "False",
            "PROMPT_FORMAT": "chat_system_user",
            "MODEL_NAME": "meta-llama/Llama-3.1-8B-Instruct",
            "MAX_NEW_TOKENS": "20",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }.items():
            assert env[name] == expected, (case["name"], name, env[name])
        assert config["eval_script"] == "run/eval.sh"
        assert config["eval_cases"] == [{"TOP_K": "20", "EVAL_BSZ": "1"}]
        db = Path(env["DB_DIR"])
        assert str(db) == case["db_dir"]
        assert (db / "chroma.sqlite3").is_file()
        assert (db / "subchunks.json").is_file()
        manifest_path = db / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["retrievable_chunk_size"] == 512
        assert manifest["tokenizer_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        if case["db_kind"] == "fixed":
            assert manifest["splitter"] == "fixed_size"
            assert manifest["max_subchunk_tokens"] is None
        else:
            assert manifest["splitter"] == "sentence"
            assert manifest["max_subchunk_tokens"] == 100
            assert manifest["max_subchunk_tokenizer_name"] == "lightonai/LateOn"
        query_path = Path(env["QUERY_FILE"])
        answer_path = Path(env["ANSWER_FILE"])
        queries = [json.loads(line) for line in query_path.open() if line.strip()]
        answers = [json.loads(line) for line in answer_path.open() if line.strip()]
        assert len(queries) == len(answers) == case["query_count"]
        assert all(q["id"] == a["id"] for q, a in zip(queries, answers))
        checked.append(
            {
                **case,
                "config_sha256": digest(Path(case["config"])),
                "db_manifest_sha256": digest(manifest_path),
                "query_sha256": digest(query_path),
                "answer_sha256": digest(answer_path),
                "db_settings": manifest,
            }
        )
    (OUT / "configuration_validation.json").write_text(
        json.dumps(checked, ensure_ascii=False, indent=2) + "\n"
    )

    os.environ.update(config["fixed_env"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(REPO / "src"))
    import torch
    from transformers import AutoTokenizer

    from compressor.factory import compress_docs
    from engine import QueryProcessor
    from model import LLMModel
    from prompt import PromptProcessor
    from vectordb import ChromaDB

    torch.set_num_threads(32)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True
    )
    prompt_processor = PromptProcessor(
        tokenizer, LLMModel.SYSTEM_PROMPT, "chat_system_user"
    )
    samples = []
    for case in preparation["cases"]:
        print("Checking native retrieval and prompt:", case["name"], flush=True)
        query = json.loads(next(Path(case["query_file"]).open()))
        vectordb = ChromaDB(case["db_dir"])
        batch_docs = vectordb.find_top_k_docs(top_k=20, queries=[query["query"]])
        assert len(batch_docs) == 1 and len(batch_docs[0]) == 20
        assert compress_docs([query["query"]], batch_docs, option=None) is batch_docs
        docs = batch_docs[0]
        expected = [unit.text for doc in docs for unit in doc.cacheables]
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.model = SimpleNamespace(prompt_processor=prompt_processor)
        assert processor._extract_passages(docs) == expected
        prompt = processor.concatenate_query_and_doc(docs, query["query"])
        assert prompt == prompt_processor.build_cache_aligned_qa_prompt(
            query=query["query"], passages=expected
        )
        if case["db_kind"] == "fixed":
            assert all(len(doc.cacheables) == 1 for doc in docs)
            assert expected == [doc.text for doc in docs]
        sample = {
            "name": case["name"],
            "query": query,
            "db_dir": vectordb.db_dir,
            "retrieved_chunks": len(docs),
            "cacheables_per_chunk": [len(doc.cacheables) for doc in docs],
            "prompt_passages": expected,
            "prompt": prompt,
            "full_input_tokens": len(tokenizer.encode(prompt)),
            "compression_identity_verified": True,
        }
        samples.append(sample)
        (OUT / "native_prompt_validation.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n"
        )
        print(case["name"], "PASS", sample["full_input_tokens"], flush=True)


if __name__ == "__main__":
    main()
