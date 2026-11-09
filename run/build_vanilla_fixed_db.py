"""Reuse the historical fixed-token splitter with the HEAD Chroma storage adapter."""

import argparse
import ast
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Any, List


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def legacy_splitter(cfg, tokenizer):
    from chunk import CacheableChunk

    def cacheable(**kwargs):
        kwargs.pop("chunk_size", None)
        kwargs.pop("prompt_tokenizer_name", None)
        return CacheableChunk(**kwargs)

    source = Path(cfg["legacy_splitter_source"]).read_text()
    parsed = ast.parse(source)
    wanted = {"SplitDocumentResult", "DocumentSplitter", "FixedSizeSplitter"}
    module = ast.Module(
        body=[
            n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in wanted
        ],
        type_ignores=[],
    )
    assert {n.name for n in module.body} == wanted
    namespace = {
        "os": os,
        "dataclass": dataclass,
        "List": List,
        "CacheableChunk": cacheable,
        "RetrievableChunk": Any,
        "UnitParser": Any,
    }
    exec(compile(module, cfg["legacy_splitter_source"], "exec"), namespace)
    overhead = len(tokenizer.encode("\n\n", add_special_tokens=False))
    assert overhead == 1
    splitter = namespace["FixedSizeSplitter"](
        docs_dir=cfg["docs"],
        model=SimpleNamespace(tokenizer=tokenizer, model_name=cfg["tokenizer"]),
        cacheable_chunk_size=cfg["chunk_size"],
        retrievable_chunk_size=cfg["chunk_size"],
        content_chunk_size=cfg["chunk_size"] - overhead,
    )
    return splitter


def prepare_covid(cfg):
    source = Path(cfg["covid_source"])
    records = [json.loads(line) for line in source.open() if line.strip()]
    texts = {}
    for record in records:
        key = str(record["document_id"])
        if key in texts:
            assert texts[key] == record["context"], key
        else:
            texts[key] = record["context"]
    docs = Path(cfg["docs"])
    docs.mkdir(parents=True, exist_ok=True)
    for key, text in texts.items():
        p = docs / (key + ".txt")
        if p.exists():
            assert p.read_text() == text
        else:
            p.write_text(text)
    assert len(list(docs.glob("*.txt"))) == len(texts)
    save(
        Path(cfg["output"]) / "covid_documents.json",
        {
            "scope": "custom shared corpus of all original COVID-QA contexts grouped by existing document_id; no context rewriting",
            "source": str(source),
            "questions": len(records),
            "documents": len(texts),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )


def migrate_existing(cfg, collection):
    from chunk import CacheableChunk
    from subchunk_store import SubchunkWriter

    db = Path(cfg["db"])
    output = Path(cfg["output"])
    count = collection.count()
    con = sqlite3.connect((db / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True)
    ready = con.execute(
        "SELECT count(*) FROM embedding_metadata WHERE key='subchunk_start'"
    ).fetchone()[0]
    if (db / "subchunks.json").exists() and ready == count:
        con.close()
        return
    assert (
        con.execute(
            "SELECT count(*) FROM embedding_metadata WHERE key='cacheables_json'"
        ).fetchone()[0]
        == count
    )
    tmp = db / "subchunks.json.tmp"
    if tmp.exists():
        tmp.unlink()
    writer = SubchunkWriter(db)
    last = -1
    done = 0
    try:
        while True:
            rows = con.execute(
                "SELECT e.id,e.embedding_id,m.string_value,d.string_value FROM embeddings e JOIN embedding_metadata m ON m.id=e.id AND m.key='cacheables_json' JOIN embedding_metadata d ON d.id=e.id AND d.key='chroma:document' WHERE e.id>? ORDER BY e.id LIMIT 5461",
                (last,),
            ).fetchall()
            if not rows:
                break
            ids = []
            metadata = []
            for iid, cid, raw, text in rows:
                payload = json.loads(raw)
                assert len(payload) == 1 and payload[0]["text"] == text, cid
                item = payload[0]
                assert type(item["prompt_token_count"]) is int
                start = writer.count
                writer.add_document([CacheableChunk.from_payload(item)])
                ids.append(cid)
                metadata.append({"subchunk_start": start, "subchunk_end": writer.count})
                last = iid
            collection.update(ids=ids, metadatas=metadata)
            done += len(ids)
            save(
                output / "status.json",
                {
                    "stage": "legacy_metadata_compatibility",
                    "records": done,
                    "total": count,
                    "embeddings_recomputed": False,
                },
            )
            print("legacy_metadata_compatibility", done, count, flush=True)
        assert done == count
        writer.finish()
    finally:
        writer.close()
        con.close()
    assert collection.count() == count
    save(
        output / "metadata_compatibility.json",
        {
            "records": count,
            "embeddings_recomputed": False,
            "existing_texts_ids_and_answer_metrics_unchanged": True,
            "legacy_cacheables_json_retained": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CHROMA_EMBED_BACKEND="bge_small_v1_5",
        CHROMA_EMBED_DEVICE="cuda",
        CHROMA_EMBED_BATCH_SIZE=str(cfg["embedding_batch_size"]),
    )
    sys.path.insert(0, str(Path(cfg["code"]) / "src"))
    db = Path(cfg["db"])
    import chromadb

    if (db / "chroma.sqlite3").exists():
        client = chromadb.PersistentClient(path=str(db))
        collection = client.get_collection("doc_collection", embedding_function=None)
        if (db / "build_manifest.json").exists():
            manifest = json.loads((db / "build_manifest.json").read_text())
            assert (
                manifest["splitter"] == "fixed_size"
                and manifest["cacheable_chunk_size"] == cfg["chunk_size"]
                and manifest["retrievable_chunk_size"] == cfg["chunk_size"]
            )
            assert (
                manifest["embedding_backend"] == "bge_small_v1_5"
                and manifest["tokenizer_name"] == cfg["tokenizer"]
            )
            assert collection.count() > 0
            migrate_existing(cfg, collection)
            save(
                output / "complete.json",
                {
                    "stage": "reused_existing_db",
                    "entries": collection.count(),
                    "db": str(db),
                    "encoding_skipped": True,
                },
            )
            save(
                db.parent / "vanilla_ready_20261028.json",
                {
                    "db": str(db),
                    "entries": collection.count(),
                    "chunk_size": cfg["chunk_size"],
                    "existing_db_reused": True,
                },
            )
            print("Reused existing DB", db, collection.count(), flush=True)
            return
        del collection, client
    if cfg.get("covid_source"):
        prepare_covid(cfg)
    from transformers import AutoTokenizer, set_seed
    import torch

    torch.set_num_threads(32)
    set_seed(123)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True)
    splitter = legacy_splitter(cfg, tokenizer)
    build = db.parent / "build"
    build.mkdir(parents=True, exist_ok=True)
    stage = build / "fixed_chunks.sqlite3"
    con = sqlite3.connect(stage)
    con.execute(
        "CREATE TABLE IF NOT EXISTS chunks(row INTEGER PRIMARY KEY,id TEXT UNIQUE,parent TEXT,start INTEGER,end INTEGER,text TEXT,tokens INTEGER)"
    )
    con.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value INTEGER)")
    order = Path(cfg["document_order"])
    if order.exists():
        files = json.loads(order.read_text())
    else:
        files = os.listdir(cfg["docs"])
        save(order, files)
    prior = con.execute("SELECT value FROM state WHERE key='documents'").fetchone()
    begin = prior[0] if prior else 0
    row = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    started = time.monotonic()
    for ordinal in range(begin, len(files)):
        name = files[ordinal]
        text = (Path(cfg["docs"]) / name).read_text()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        units, _, _, _ = splitter.build_chunks(name, text, tokens)
        for unit in units:
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
                (
                    row,
                    unit.id,
                    name,
                    unit.chunk_start,
                    unit.chunk_end,
                    unit.text,
                    unit.prompt_token_count,
                ),
            )
            row += 1
        con.execute(
            "INSERT OR REPLACE INTO state VALUES('documents',?)", (ordinal + 1,)
        )
        if (ordinal + 1) % 1000 == 0:
            con.commit()
            save(
                output / "status.json",
                {
                    "stage": "fixed_chunk_preparation",
                    "documents": ordinal + 1,
                    "total_documents": len(files),
                    "chunks": row,
                    "seconds": time.monotonic() - started,
                },
            )
            print("fixed_chunk_preparation", ordinal + 1, len(files), row, flush=True)
    con.commit()
    total = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print("fixed_chunk_preparation_complete", len(files), total, flush=True)
    from chunk import CacheableChunk, RetrievableChunk
    from subchunk_store import SubchunkWriter

    db.mkdir(parents=True, exist_ok=True)
    if not (db / "subchunks.json").exists():
        tmp = db / "subchunks.json.tmp"
        if tmp.exists():
            tmp.unlink()
        writer = SubchunkWriter(db)
        try:
            for text, count in con.execute(
                "SELECT text,tokens FROM chunks ORDER BY row"
            ):
                writer.add_document(
                    [
                        CacheableChunk(
                            id=writer.count, text=text, prompt_token_count=count
                        )
                    ]
                )
            assert writer.count == total
            writer.finish()
        finally:
            writer.close()
    # Recheck availability after CPU preparation, before loading the build encoder.
    import subprocess

    while True:
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        if not [p for p in pids if p.strip() and int(p) != os.getpid()]:
            break
        save(output / "status.json", {"stage": "waiting_for_gpu", "pids": pids})
        time.sleep(30)
    from vectordb import ChromaDB

    vector_db = ChromaDB.for_build(
        str(db),
        embedding_device="cuda",
        embedding_batch_size=cfg["embedding_batch_size"],
    )
    progress = output / "encoding_progress.json"
    done = json.loads(progress.read_text())["encoded"] if progress.exists() else 0
    assert vector_db.db.count() == done
    batch = cfg["db_batch_size"]
    while done < total:
        rows = con.execute(
            "SELECT row,id,parent,start,end,text,tokens FROM chunks WHERE row>=? ORDER BY row LIMIT ?",
            (done, batch),
        ).fetchall()
        records = []
        for rid, cid, parent, start, end, text, count in rows:
            unit = CacheableChunk(
                id=rid,
                text=text,
                parent_doc_id=parent,
                chunk_start=start,
                chunk_end=end,
                prompt_token_count=count,
            )
            records.append(
                RetrievableChunk(
                    id=cid,
                    text=text,
                    cacheables=[unit],
                    chunk_size=cfg["chunk_size"],
                    token_count=end - start,
                    cache_unit="token",
                    metadata={
                        "parent_doc_id": parent,
                        "source_token_start": start,
                        "source_token_end": end,
                    },
                )
            )
        vector_db.store(records)
        done += len(rows)
        value = {
            "stage": "encoding_and_indexing",
            "encoded": done,
            "total": total,
            "seconds": time.monotonic() - started,
        }
        save(progress, value)
        save(output / "status.json", value)
        print(json.dumps(value), flush=True)
    assert vector_db.db.count() == total
    with sqlite3.connect(
        (db / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as check:
        legacy_rows = check.execute(
            "SELECT count(*) FROM embedding_metadata WHERE key='cacheables_json'"
        ).fetchone()[0]
    if legacy_rows:
        raise RuntimeError(
            f"New vanilla DB must not store cacheables_json; found {legacy_rows} rows"
        )
    from materialize.db_manifest import build_db_manifest, write_db_build_manifest

    manifest = build_db_manifest(
        splitter="fixed_size",
        cacheable_chunk_size=cfg["chunk_size"],
        retrievable_chunk_size=cfg["chunk_size"],
        max_subchunk_tokens=None,
        tokenizer_name=cfg["tokenizer"],
        deduplicate_documents_by_hash=False,
        embedding_backend="bge_small_v1_5",
        db_batch_size=cfg["db_batch_size"],
        embedding_device="cuda",
        embedding_batch_size=cfg["embedding_batch_size"],
    )
    manifest.update(
        fixed_splitter_source_commit="1b52bc8",
        content_chunk_size=cfg["chunk_size"] - 1,
        visible_suffix_tokens=1,
        subchunk_storage="row-indexed text/count list",
        scope=cfg["scope"],
    )
    write_db_build_manifest(db, manifest)
    save(
        output / "complete.json",
        {
            "stage": "built",
            "db": str(db),
            "documents": len(files),
            "entries": total,
            "chunk_size": cfg["chunk_size"],
            "body_tokens_per_full_chunk": cfg["chunk_size"] - 1,
            "seconds": time.monotonic() - started,
        },
    )
    save(
        db.parent / "vanilla_ready_20261028.json",
        {
            "db": str(db),
            "entries": total,
            "chunk_size": cfg["chunk_size"],
            "existing_db_reused": False,
        },
    )
    con.close()
    stage.unlink()
    print("DB complete", db, total, flush=True)


if __name__ == "__main__":
    main()
