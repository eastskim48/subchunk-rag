"""Index existing BGE region vectors and IDs without duplicating their payloads."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_bge_direct_regions import read, save, sha


def open_inputs(spec):
    import numpy as np

    artifact = Path(spec["artifact_dir"])
    prepared, encoded = read(artifact / "prepared.json"), read(
        artifact / "encoded.json"
    )
    assert Path(prepared["source_db"]).resolve() == Path(spec["source_db"]).resolve()
    assert prepared["region_count"] == encoded["region_count"]
    vector_file = artifact / encoded["embedding_file"]
    assert sha(vector_file) == encoded["embedding_sha256"]
    vectors = np.memmap(
        vector_file,
        mode="r",
        dtype="<f4",
        shape=(encoded["region_count"], encoded["embedding_dim"]),
    )
    store = artifact / "regions.sqlite3"
    connection = sqlite3.connect(store.resolve().as_uri() + "?mode=ro", uri=True)
    assert connection.execute("select count(*) from regions").fetchone()[0] == len(
        vectors
    )
    return prepared, encoded, vectors, connection


def index(spec, batch_size, sync_threshold):
    import chromadb
    import numpy as np
    from vectordb import ChromaDB

    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index_built.json").exists():
        return
    prepared, encoded, vectors, connection = open_inputs(spec)
    configuration = {"hnsw": dict(ChromaDB.DEFAULT_COLLECTION_CONFIGURATION["hnsw"])}
    configuration["hnsw"]["sync_threshold"] = sync_threshold
    client = chromadb.PersistentClient(path=str(out / "db"))
    collection = client.get_or_create_collection(
        "doc_collection", embedding_function=None, configuration=configuration
    )
    assert collection.configuration["hnsw"]["sync_threshold"] == sync_threshold
    progress = out / "index_progress.json"
    previous = read(progress) if progress.exists() else {}
    done = previous.get("completed", 0)
    resumed_from = done
    previous_seconds = previous.get("elapsed_sec", 0.0)
    batch_size = min(batch_size, client.get_max_batch_size())
    assert 0 <= done <= collection.count() <= min(len(vectors), done + batch_size)
    started = time.monotonic()
    while done < len(vectors):
        rows = connection.execute(
            "select row_id,region_id from regions where row_id>=? order by row_id limit ?",
            (done, batch_size),
        ).fetchall()
        assert [r[0] for r in rows] == list(range(done, done + len(rows)))
        collection.upsert(
            ids=[r[1] for r in rows],
            embeddings=np.asarray(vectors[done : done + len(rows)]),
        )
        done += len(rows)
        state = dict(
            stage="index",
            completed=done,
            total=len(vectors),
            elapsed_sec=previous_seconds + time.monotonic() - started,
            session_elapsed_sec=time.monotonic() - started,
            resumed_from=resumed_from,
        )
        save(progress, state)
        print(json.dumps(state), flush=True)
    assert collection.count() == len(vectors)
    connection.close()
    manifest = dict(
        {**prepared, **encoded},
        format="full_corpus_direct_bge_vectors_v1",
        storage_layout="vectors_and_ids_only",
        collection="doc_collection",
        region_store=str((Path(spec["artifact_dir"]) / "regions.sqlite3").resolve()),
        source_artifact=str(Path(spec["artifact_dir"]).resolve()),
        index_entries=len(vectors),
        hnsw_configuration=collection.configuration,
        index_seconds=previous_seconds + time.monotonic() - started,
        reused_existing_embeddings=True,
    )
    save(out / "index_built.json", manifest)


def verify(spec):
    """Run in a fresh process before publishing a completion marker."""
    import chromadb
    import numpy as np
    from subchunk_store import load_subchunks

    out = Path(spec["output"])
    if (out / "index_complete.json").exists():
        return
    started = time.monotonic()
    manifest = read(out / "index_built.json")
    prepared, encoded, vectors, connection = open_inputs(spec)
    assert encoded["embedding_sha256"] == manifest["embedding_sha256"]
    collection = chromadb.PersistentClient(path=str(out / "db")).get_collection(
        "doc_collection", embedding_function=None
    )
    assert collection.count() == manifest["index_entries"] == len(vectors)
    units = load_subchunks(spec["source_db"])

    def verify_mapping(region_id):
        row = connection.execute(
            "select row_id,text,metadata from regions where region_id=?", (region_id,)
        ).fetchone()
        assert row is not None, region_id
        meta = json.loads(row[2])
        assert meta["region_id"] == region_id
        assert row[1] == " ".join(
            units[i][0] for i in range(meta["subchunk_start"], meta["subchunk_end"])
        )
        return row[0]

    samples = sorted(
        set(np.linspace(0, len(vectors) - 1, min(64, len(vectors)), dtype=int))
    )
    sample_ids = [
        connection.execute(
            "select region_id from regions where row_id=?", (int(i),)
        ).fetchone()[0]
        for i in samples
    ]
    stored = collection.get(
        ids=sample_ids, include=["documents", "metadatas", "embeddings"]
    )
    assert set(stored["ids"]) == set(sample_ids)
    assert all(value is None for value in stored["documents"])
    assert all(value is None for value in stored["metadatas"])
    for region_id, vector in zip(stored["ids"], stored["embeddings"], strict=True):
        row_id = verify_mapping(region_id)
        np.testing.assert_allclose(
            vector, vectors[row_id], rtol=4 * np.finfo(np.float32).eps, atol=0
        )
    search_ids = []
    for row_id in sorted({0, len(vectors) // 2, len(vectors) - 1}):
        hits = collection.query(
            query_embeddings=[vectors[row_id].tolist()],
            n_results=min(3, len(vectors)),
            include=["distances"],
        )
        assert len(hits["ids"][0]) == min(3, len(vectors))
        assert np.isfinite(hits["distances"][0]).all()
        for region_id in hits["ids"][0]:
            verify_mapping(region_id)
            search_ids.append(region_id)
    connection.close()
    sql = sqlite3.connect(
        (out / "db/chroma.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    assert sql.execute("select count(*) from embedding_metadata").fetchone()[0] == 0
    assert (
        sql.execute("select count(*) from embedding_fulltext_search").fetchone()[0] == 0
    )
    sql.close()
    manifest["verification"] = dict(
        sampled_vectors=len(samples),
        mapped_search_hits=len(search_ids),
        metadata_rows=0,
        fulltext_rows=0,
        seconds=time.monotonic() - started,
    )
    save(out / "db/build_manifest.json", manifest)
    save(out / "index_complete.json", manifest)
    print(json.dumps(dict(stage="verified", entries=len(vectors))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", choices=["index", "verify"], required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    spec = cfg["datasets"][args.dataset]
    if args.phase == "index":
        index(spec, cfg["index_batch_size"], cfg["index_sync_threshold"])
    else:
        verify(spec)


if __name__ == "__main__":
    main()
