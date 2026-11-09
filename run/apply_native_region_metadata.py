"""Apply previously materialized native-tokenizer regions without re-encoding sentences."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--lateon-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "complete.json"
    if marker.exists():
        print("Region update already complete", flush=True)
        return
    begun = time.monotonic()
    info = json.loads((args.regions_dir / "regions_complete.json").read_text())
    assert info["tokenizer"] == "lightonai/LateOn" and info["window"] == 180
    assert info["scope"] == "full existing DB"
    con = sqlite3.connect(
        (args.regions_dir / "regions.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    specs = {
        cid: (json.loads(a), json.loads(b))
        for cid, a, b in con.execute("SELECT id,starts,ends FROM chunks")
    }
    assert len(specs) == info["chunks"]
    units = json.loads((args.db_dir / "subchunks.json").read_text())
    source = sqlite3.connect(
        (args.db_dir / "chroma.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    bounds = {
        cid: (a, b)
        for cid, a, b in source.execute(
            "SELECT e.embedding_id,a.int_value,b.int_value FROM embeddings e JOIN embedding_metadata a ON a.id=e.id AND a.key='subchunk_start' JOIN embedding_metadata b ON b.id=e.id AND b.key='subchunk_end'"
        )
    }
    source.close()
    assert specs.keys() == bounds.keys()
    count = 0
    for rid, text in con.execute("SELECT id,text FROM regions ORDER BY rowid"):
        cid, ordinal = rid.rsplit("::dense_sliding_region_", 1)
        center = int(ordinal.split("::", 1)[0])
        starts, ends = specs[cid]
        start, end = starts[center], ends[center]
        a, b = bounds[cid]
        assert 0 <= start < end <= b - a
        assert " ".join(units[i][0] for i in range(a + start, a + end)) == text, rid
        count += 1
    assert count == info["regions"]
    con.close()
    print(
        json.dumps(
            {
                "stage": "validated_native_regions",
                "chunks": len(specs),
                "regions": count,
            }
        ),
        flush=True,
    )
    import chromadb

    collection = chromadb.PersistentClient(
        path=str(args.db_dir.resolve())
    ).get_collection("doc_collection", embedding_function=None)
    assert collection.count() == len(specs)
    old_common = collection.metadata or {}
    (args.output / "previous_collection_metadata.json").write_text(
        json.dumps(old_common, indent=2) + "\n"
    )
    # Only region metadata is replaced; IDs, source bounds, text and vectors stay intact.
    collection.modify(metadata={**old_common, "regions_ready": False})
    backup = args.output / "previous_region_metadata.jsonl"
    done = 0
    with backup.open("w") as f:
        for offset in range(0, len(specs), 2048):
            rows = collection.get(limit=2048, offset=offset, include=["metadatas"])
            updates = []
            for cid, meta in zip(rows["ids"], rows["metadatas"], strict=True):
                assert (
                    tuple([meta["subchunk_start"], meta["subchunk_end"]]) == bounds[cid]
                )
                f.write(
                    json.dumps(
                        {
                            "id": cid,
                            "region_starts": meta.get("region_starts"),
                            "region_ends": meta.get("region_ends"),
                        }
                    )
                    + "\n"
                )
                a, b = specs[cid]
                updates.append({**meta, "region_starts": a, "region_ends": b})
            collection.update(ids=rows["ids"], metadatas=updates)
            done += len(updates)
            print(
                json.dumps(
                    {"stage": "apply_regions", "chunks": done, "total": len(specs)}
                ),
                flush=True,
            )
    collection.modify(
        metadata={
            **old_common,
            "regions_ready": True,
            "region_token_budget": 180,
            "region_tokenizer_name": info["tokenizer"],
            "region_tokenizer_revision": info["revision"],
        }
    )
    verified = 0
    for offset in range(0, len(specs), 2048):
        rows = collection.get(limit=2048, offset=offset, include=["metadatas"])
        for cid, meta in zip(rows["ids"], rows["metadatas"], strict=True):
            a, b = specs[cid]
            assert meta["region_starts"] == a and meta["region_ends"] == b
            assert tuple([meta["subchunk_start"], meta["subchunk_end"]]) == bounds[cid]
            verified += 1
    assert verified == len(specs)
    if args.lateon_artifact is not None:
        index_path = args.lateon_artifact / "data/index.json"
        data = json.loads(index_path.read_text())
        (args.output / "previous_lateon_data_index.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
        data.update(
            region_spec_count=count,
            region_spec_chunk_count=len(specs),
            region_membership_source="chroma:doc_collection",
            region_tokenizer_name=info["tokenizer"],
            region_tokenizer_revision=info["revision"],
        )
        index_path.write_text(json.dumps(data, indent=2) + "\n")
    result = {
        "chunks": len(specs),
        "regions": count,
        "tokenizer": info["tokenizer"],
        "window": 180,
        "verified_chunks": verified,
        "sentence_embedding_payload_unchanged": True,
        "retrieval_text_and_vectors_updated": False,
        "seconds": time.monotonic() - begun,
        "region_sqlite_sha256": digest(args.regions_dir / "regions.sqlite3"),
    }
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
