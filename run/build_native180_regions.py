"""Materialize shared native-LateOn region definitions and DenseOn vectors."""

import argparse
from collections import defaultdict
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def regions(cfg):
    import numpy as np
    from transformers import AutoTokenizer
    from encoder.colbert import ColBERTEncoder
    from encoder.lateon import LATEON_MODEL, LATEON_REVISION
    from materialize.colbert_materializer import (
        ColBERTWindowEncoder,
        _window_bounded_region_index_specs,
    )
    from subchunk_store import pack_regions

    folder = Path(cfg["artifacts"])
    if (folder / "regions_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    wanted = None
    coverage = set()
    if cfg.get("preparation_snapshot"):
        wanted = set()
        with open(cfg["preparation_snapshot"]) as f:
            for line in f:
                for doc in json.loads(line)["docs"]:
                    wanted.add(str(doc["id"]))
                    coverage.update(int(c["id"]) for c in doc["cacheables"])
        np.save(
            folder / "lateon_coverage.npy", np.array(sorted(coverage), dtype=np.int64)
        )
    source_db = Path(cfg["db"])
    source = sqlite3.connect(
        (source_db / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    query = """SELECT e.embedding_id,p.string_value,s.int_value,t.int_value
    FROM embedding_metadata p JOIN embeddings e ON e.id=p.id
    JOIN embedding_metadata s ON s.id=p.id AND s.key='subchunk_start'
    JOIN embedding_metadata t ON t.id=p.id AND t.key='subchunk_end'
    JOIN segments g ON g.id=e.segment_id JOIN collections c ON c.id=g.collection
    WHERE p.key='parent_doc_id' AND c.name='doc_collection'
    ORDER BY p.string_value,s.int_value,t.int_value"""
    grouped = defaultdict(list)
    needed_parents = set()
    for cid, parent, start, end in source.execute(query):
        grouped[parent].append((cid, start, end))
        if wanted is None or cid in wanted:
            needed_parents.add(parent)
    source.close()
    print("loading saved sentence text", flush=True)
    units = json.load(open(source_db / "subchunks.json"))
    tok = AutoTokenizer.from_pretrained(
        LATEON_MODEL, revision=LATEON_REVISION, local_files_only=True
    )
    tok.model_max_length = 10**9
    builder = SimpleNamespace(
        doc_tokenizer=tok,
        encoder_impl="lateon_transformers",
        token_counts_without_specials=None,
        doc_token_overhead=len(tok("", add_special_tokens=True)["input_ids"]) + 1,
        _next_window_candidate=ColBERTWindowEncoder._next_window_candidate,
    )
    builder.token_counts_without_specials = (
        lambda texts: ColBERTEncoder.token_counts_without_specials(builder, texts)
    )
    target = folder / "regions.sqlite3"
    staging = folder / "regions.building.sqlite3"
    if staging.exists():
        staging.unlink()
    con = sqlite3.connect(staging)
    con.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, starts TEXT NOT NULL, ends TEXT NOT NULL)"
    )
    con.execute(
        "CREATE TABLE regions (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, text TEXT NOT NULL)"
    )
    chunks = total = 0
    begun = time.monotonic()
    for ordinal, parent in enumerate(sorted(needed_parents), 1):
        bounds = grouped[parent]
        ids = sorted({i for _, s, e in bounds for i in range(s, e) if units[i][0]})
        texts = [units[i][0] for i in ids]
        windows = ColBERTWindowEncoder.build_centered_windows(builder, texts, 180)
        maps = {
            str(i): [str(ids[j]) for j in spec.selected_indices]
            for i, spec in zip(ids, windows, strict=True)
        }
        for cid, start, end in bounds:
            if wanted is not None and cid not in wanted:
                continue
            chunk_ids = [str(i) for i in range(start, end)]
            specs = _window_bounded_region_index_specs(chunk_ids, maps)
            packed = pack_regions(specs)
            con.execute(
                "INSERT INTO chunks VALUES (?,?,?)",
                (
                    cid,
                    json.dumps(packed["region_starts"]),
                    json.dumps(packed["region_ends"]),
                ),
            )
            for center, (_, indices) in enumerate(specs):
                text = " ".join(units[start + i][0] for i in indices)
                con.execute(
                    "INSERT INTO regions VALUES (?,?,?)",
                    (
                        total,
                        f"{cid}::dense_sliding_region_{center}::direct_dense",
                        text,
                    ),
                )
                total += 1
            chunks += 1
        if ordinal % 1000 == 0:
            con.commit()
            state = dict(
                stage="region_definitions",
                parents=ordinal,
                total_parents=len(needed_parents),
                chunks=chunks,
                regions=total,
                seconds=time.monotonic() - begun,
            )
            save(folder / "regions_progress.json", state)
            print(json.dumps(state), flush=True)
    con.commit()
    con.close()
    staging.replace(target)
    if wanted is not None:
        assert chunks == len(wanted)
    save(
        folder / "regions_complete.json",
        dict(
            tokenizer=LATEON_MODEL,
            revision=LATEON_REVISION,
            window=180,
            token_overhead=builder.doc_token_overhead,
            chunks=chunks,
            regions=total,
            scope=(
                "existing snapshot chunk union"
                if wanted is not None
                else "full existing DB"
            ),
            preparation_only=True,
        ),
    )


def dense(cfg):
    import numpy as np
    import torch
    from encoder.dense import DenseTextEmbedder, DENSEON_MODEL, default_passage_prefix

    folder = Path(cfg["artifacts"])
    if (folder / "dense_complete.json").exists():
        return
    info = json.load(open(folder / "regions_complete.json"))
    total = info["regions"]
    con = sqlite3.connect((folder / "regions.sqlite3").as_uri() + "?mode=ro", uri=True)
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
        other = [
            int(p.strip()) for p in pids if p.strip() and int(p.strip()) != os.getpid()
        ]
        if not other:
            break
        print("Waiting for GPU", other, flush=True)
        time.sleep(30)
    encoder = DenseTextEmbedder(DENSEON_MODEL, device="cpu", batch_size=128)
    encoder.model.to("cuda")
    encoder.device = "cuda"
    assert next(encoder.model.parameters()).dtype == torch.float32
    # Probe the longest native tokenized region, respecting the encoder's own cap.
    longest = con.execute(
        "SELECT text FROM regions ORDER BY length(text) DESC LIMIT 1"
    ).fetchone()[0]
    prefix = default_passage_prefix(DENSEON_MODEL)
    batch = 128
    trials = []
    while batch <= min(total, 4096):
        encoder.batch_size = batch
        try:
            v = encoder.embed_texts([prefix + longest] * batch)
            assert torch.isfinite(v).all()
            del v
            trials.append(dict(batch=batch, fits=True))
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch=batch, fits=False))
            torch.cuda.empty_cache()
            break
    batch = max(t["batch"] for t in trials if t["fits"])
    encoder.batch_size = batch
    save(folder / "dense_batch.json", dict(selected_batch=batch, trials=trials))
    path = folder / "dense_vectors.npy"
    vectors = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=(total, 768)
    )
    progress = folder / "dense_progress.json"
    start = json.load(progress.open())["encoded"] if progress.exists() else 0
    begun = time.monotonic()
    while start < total:
        rows = con.execute(
            "SELECT text FROM regions WHERE rowid>=? ORDER BY rowid LIMIT ?",
            (start, batch),
        ).fetchall()
        try:
            v = encoder.embed_texts([prefix + r[0] for r in rows])
        except torch.cuda.OutOfMemoryError:
            batch //= 2
            assert batch > 0
            encoder.batch_size = batch
            torch.cuda.empty_cache()
            continue
        assert torch.isfinite(v).all() and torch.isfinite(v.to(torch.float16)).all()
        vectors[start : start + len(rows)] = v.to(torch.float16).numpy()
        vectors.flush()
        start += len(rows)
        state = dict(
            stage="dense_embedding",
            encoded=start,
            total=total,
            batch=batch,
            seconds=time.monotonic() - begun,
        )
        save(progress, state)
        print(json.dumps(state), flush=True)
    con.close()
    save(
        folder / "dense_complete.json",
        dict(
            model=DENSEON_MODEL,
            compute_dtype="float32",
            storage_dtype="float16",
            rows=total,
            dimensions=768,
            document_prefix=prefix,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["regions", "dense"], required=True)
    args = parser.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    cfg = json.load(args.config.open())
    {"regions": regions, "dense": dense}[args.action](cfg)
