"""Build a full-corpus direct region index from the native stored CASS regions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def region_units(chunk_id, metadata, subchunks):
    from compressor.methods.dense import DenseSlidingSubchunkCompressor
    from subchunk_store import cacheables_from_metadata, unpack_regions

    cacheables = cacheables_from_metadata(metadata, subchunks)
    source = [unit for unit in cacheables if unit.text]
    specs = unpack_regions(metadata, [unit.id for unit in source])
    for center, positions in specs:
        unit = DenseSlidingSubchunkCompressor._direct_subchunk_cacheable(
            dict(
                region_id=f"{chunk_id}::dense_sliding_region_{center}",
                parent_doc_id=metadata["parent_doc_id"],
                source_cacheables=source,
                selected_indices=positions,
            )
        )
        ids = [int(sid) for sid in unit.sentence_ids]
        assert ids == list(range(ids[0], ids[-1] + 1))
        yield unit, dict(
            parent_doc_id=metadata["parent_doc_id"],
            chunk_id=str(chunk_id),
            region_id=str(unit.id),
            subchunk_start=ids[0],
            subchunk_end=ids[-1] + 1,
        )


def prepare(spec):
    from materialize.db_cacheables import iter_db_records
    from subchunk_store import load_subchunks

    source = Path(spec["db"])
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "prepared.json"
    hashes = {
        name: sha(source / name) for name in ["build_manifest.json", "subchunks.json"]
    }
    if manifest.exists():
        assert read(manifest)["source_hashes"] == hashes
        return
    temporary = out / "regions.build.sqlite3"
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute(
        "create table regions (row_id integer primary key, region_id text unique, "
        "text text not null, metadata text not null)"
    )
    subchunks = load_subchunks(str(source.resolve()))
    import chromadb

    collection = chromadb.PersistentClient(path=str(source)).get_collection(
        "doc_collection", embedding_function=None
    )
    assert collection.metadata["regions_ready"]
    assert int(collection.metadata["region_token_budget"]) == 180
    started, chunks, regions = time.monotonic(), 0, 0
    for chunk_id, metadata in iter_db_records(source):
        for unit, attributes in region_units(chunk_id, metadata, subchunks):
            connection.execute(
                "insert into regions values (?,?,?,?)",
                (regions, str(unit.id), unit.text, json.dumps(attributes)),
            )
            regions += 1
        chunks += 1
        if chunks % 2048 == 0:
            connection.commit()
            progress = dict(
                stage="prepare",
                chunks=chunks,
                regions=regions,
                elapsed_sec=time.monotonic() - started,
            )
            save(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    connection.commit()
    assert chunks == collection.count()
    assert regions == connection.execute("select count(*) from regions").fetchone()[0]
    # Compare exact IDs/texts with the earlier scorer artifact, without reusing vectors.
    import torch

    stored = torch.load(
        Path(spec["scorer_artifact"]) / "embeddings.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    ids, texts = stored["region_ids"], stored["region_texts"]
    assert len(ids) == len(texts)
    sample = (
        sorted(
            {
                i * (len(ids) - 1) // min(63, len(ids) - 1)
                for i in range(min(64, len(ids)))
            }
        )
        if len(ids) > 1
        else [0]
    )
    for i in sample:
        row = connection.execute(
            "select text from regions where region_id=?", (str(ids[i]),)
        ).fetchone()
        assert row is not None and row[0] == texts[i], ids[i]
    connection.close()
    temporary.replace(out / "regions.sqlite3")
    save(
        manifest,
        dict(
            scope="all_retrieval_chunks_in_full_source_corpus",
            source_db=str(source.resolve()),
            source_hashes=hashes,
            source_chunks=chunks,
            region_count=regions,
            region_token_budget=180,
            region_text_composition="native DenseSlidingSubchunkCompressor._direct_subchunk_cacheable",
            sampled_scorer_region_matches=len(sample),
            prepare_seconds=time.monotonic() - started,
        ),
    )
    print(json.dumps(read(manifest)), flush=True)


def encode(spec, batch_size):
    import numpy as np
    import torch
    from encoder.dense import BGE_SMALL_MODEL, DenseTextEmbedder, default_query_prefix

    out = Path(spec["output"])
    prepared = read(out / "prepared.json")
    if (out / "encoded.json").exists():
        return
    import subprocess

    while True:
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        if any(pid.strip() and pid.strip() != str(os.getpid()) for pid in pids):
            print("waiting_for_gpu", pids, flush=True)
            time.sleep(30)
            continue
        try:
            gpu_reservation = torch.empty(1, device="cuda")
            break
        except RuntimeError as error:
            if "busy or unavailable" not in str(error):
                raise
            print("waiting_for_cuda_context", flush=True)
            time.sleep(30)
    torch.manual_seed(123)
    torch.set_num_threads(8)
    encoder = DenseTextEmbedder(BGE_SMALL_MODEL, device="cuda", batch_size=batch_size)
    dim = encoder.embedding_dim
    total = prepared["region_count"]
    progress = out / "encode_progress.json"
    done = read(progress)["completed"] if progress.exists() else 0
    vectors_path = out / "embeddings.float32.bin"
    if done and (
        not vectors_path.exists() or vectors_path.stat().st_size < done * dim * 4
    ):
        raise ValueError("Committed embedding rows are missing")
    connection = sqlite3.connect(out / "regions.sqlite3")
    started = time.monotonic()
    with vectors_path.open("r+b" if vectors_path.exists() else "w+b") as handle:
        handle.truncate(done * dim * 4)
        handle.seek(done * dim * 4)
        while done < total:
            rows = connection.execute(
                "select row_id,text from regions where row_id>=? order by row_id limit ?",
                (done, batch_size),
            ).fetchall()
            assert [r[0] for r in rows] == list(range(done, done + len(rows)))
            try:
                values = encoder.embed_texts([r[1] for r in rows]).float().numpy()
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                encoder.batch_size = batch_size
                torch.cuda.empty_cache()
                print(
                    f"OOM: retry uncommitted batch with batch_size={batch_size}",
                    flush=True,
                )
                continue
            assert values.shape == (len(rows), dim) and np.isfinite(values).all()
            assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=0.003)
            handle.write(values.astype("<f4", copy=False).tobytes())
            handle.flush()
            done += len(rows)
            state = dict(
                stage="encode",
                completed=done,
                total=total,
                batch_size=batch_size,
                elapsed_sec=time.monotonic() - started,
            )
            save(progress, state)
            save(out / "progress.json", state)
            print(json.dumps(state), flush=True)
    connection.close()
    assert vectors_path.stat().st_size == total * dim * 4
    save(
        out / "encoded.json",
        dict(
            model=BGE_SMALL_MODEL,
            model_revision=getattr(encoder.model.config, "_commit_hash", None),
            query_prefix=default_query_prefix(BGE_SMALL_MODEL),
            passage_prefix="",
            region_count=total,
            embedding_dim=dim,
            storage_dtype="float32",
            compute_dtype=str(next(encoder.model.parameters()).dtype),
            embedding_file=vectors_path.name,
            embedding_sha256=sha(vectors_path),
            freshly_encoded=True,
            reused_previous_vectors=False,
            final_batch_size=batch_size,
            seed=123,
        ),
    )


def index(spec, batch_size, sync_threshold=None):
    import chromadb
    import numpy as np
    from vectordb import ChromaDB

    out = Path(spec["output"])
    if (out / "index_complete.json").exists():
        return
    encoded, prepared = read(out / "encoded.json"), read(out / "prepared.json")
    path = out / encoded["embedding_file"]
    assert sha(path) == encoded["embedding_sha256"]
    total, dim = encoded["region_count"], encoded["embedding_dim"]
    vectors = np.memmap(path, mode="r", dtype="<f4", shape=(total, dim))
    configuration = {"hnsw": dict(ChromaDB.DEFAULT_COLLECTION_CONFIGURATION["hnsw"])}
    if sync_threshold is not None:
        if sync_threshold < 3:
            raise ValueError("index_sync_threshold must be at least 3")
        configuration["hnsw"]["sync_threshold"] = sync_threshold
    client = chromadb.PersistentClient(path=str(out / "db"))
    collection = client.get_or_create_collection(
        "doc_collection",
        embedding_function=None,
        configuration=configuration,
    )
    old_configuration = collection.configuration
    # get_or_create does not update an existing collection's configuration.
    if (
        sync_threshold is not None
        and old_configuration["hnsw"]["sync_threshold"] != sync_threshold
    ):
        collection.modify(configuration={"hnsw": {"sync_threshold": sync_threshold}})
    if sync_threshold is not None:
        assert collection.configuration["hnsw"]["sync_threshold"] == sync_threshold
    connection = sqlite3.connect(out / "regions.sqlite3")
    state_path = out / "index_progress.json"
    previous = read(state_path) if state_path.exists() else {}
    done = previous.get("completed", 0)
    previous_seconds = previous.get("elapsed_sec", 0.0)
    batch_size = min(batch_size, client.get_max_batch_size())
    stored_count = collection.count()
    if not 0 <= done <= stored_count <= min(total, done + batch_size):
        raise ValueError(
            f"Index/checkpoint mismatch: checkpoint={done}, stored={stored_count}"
        )
    if done:
        for row_id in sorted({0, done // 2, done - 1}):
            row = connection.execute(
                "select region_id,text,metadata from regions where row_id=?", (row_id,)
            ).fetchone()
            stored = collection.get(
                ids=[row[0]], include=["documents", "metadatas", "embeddings"]
            )
            assert stored["documents"] == [row[1]]
            assert stored["metadatas"] == [json.loads(row[2])]
            # Cosine-index round trips can differ by a few float32 rounding
            # units after normalization; the embedding-file SHA stays exact.
            np.testing.assert_allclose(
                np.asarray(stored["embeddings"])[0],
                vectors[row_id],
                rtol=4 * np.finfo(np.float32).eps,
                atol=0,
            )
    with (out / "index_sessions.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                dict(
                    started=time.time(),
                    resumed_from=done,
                    stored_count=stored_count,
                    previous_seconds=previous_seconds,
                    previous_configuration=old_configuration,
                    configuration=collection.configuration,
                )
            )
            + "\n"
        )
    print(
        json.dumps(
            dict(
                stage="index_ready",
                resumed_from=done,
                stored_count=stored_count,
                sync_threshold=collection.configuration["hnsw"]["sync_threshold"],
            )
        ),
        flush=True,
    )
    started = time.monotonic()
    while done < total:
        rows = connection.execute(
            "select row_id,region_id,text,metadata from regions where row_id>=? order by row_id limit ?",
            (done, batch_size),
        ).fetchall()
        collection.upsert(
            ids=[r[1] for r in rows],
            documents=[r[2] for r in rows],
            metadatas=[json.loads(r[3]) for r in rows],
            embeddings=np.asarray(vectors[done : done + len(rows)]),
        )
        done += len(rows)
        state = dict(
            stage="index",
            completed=done,
            total=total,
            elapsed_sec=previous_seconds + time.monotonic() - started,
            session_elapsed_sec=time.monotonic() - started,
            resumed_from=previous.get("completed", 0),
        )
        save(state_path, state)
        save(out / "progress.json", state)
        print(json.dumps(state), flush=True)
    assert collection.count() == total
    for row_id in sorted({0, total // 2, total - 1}):
        row = connection.execute(
            "select region_id,text,metadata from regions where row_id=?", (row_id,)
        ).fetchone()
        stored = collection.get(
            ids=[row[0]], include=["documents", "metadatas", "embeddings"]
        )
        assert stored["documents"] == [row[1]] and stored["metadatas"] == [
            json.loads(row[2])
        ]
        np.testing.assert_allclose(
            np.asarray(stored["embeddings"])[0],
            vectors[row_id],
            rtol=4 * np.finfo(np.float32).eps,
            atol=0,
        )
    connection.close()
    target = out / "db/subchunks.json"
    if not target.exists():
        target.symlink_to(Path(spec["db"]).resolve() / "subchunks.json")
    assert prepared["region_count"] == encoded["region_count"]
    manifest = dict(
        {**prepared, **encoded},
        format="full_corpus_direct_bge_regions_v1",
        collection="doc_collection",
        index_entries=total,
        hnsw_configuration=collection.configuration,
        query_encoder="native DenseTextEmbedder with recorded query_prefix",
        index_seconds=previous_seconds + time.monotonic() - started,
    )
    save(out / "db/build_manifest.json", manifest)
    save(out / "index_complete.json", manifest)
    save(out / "progress.json", dict(stage="complete", entries=total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--phase", choices=["prepare", "encode", "index"], required=True
    )
    args = parser.parse_args()
    cfg = read(args.config)
    spec = cfg["datasets"][args.dataset]
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    if args.phase == "prepare":
        prepare(spec)
    elif args.phase == "encode":
        encode(spec, cfg["embedding_batch_size"])
    else:
        index(spec, cfg["index_batch_size"], cfg.get("index_sync_threshold"))


if __name__ == "__main__":
    main()
