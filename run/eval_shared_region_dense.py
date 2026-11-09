"""Materialize DenseOn on an existing LateOn sweep's exact region texts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "test"), str(ROOT / "run")]


def read(path):
    return json.loads(Path(path).read_text())


def prepare(cfg):
    from build_full_lighton_artifacts import save
    from chunk import RetrievableChunk
    from colbert_artifact import ColBERTWindowArtifact
    from contributor_region_override import RegionOverride
    from encoder.colbert import ColBERTEncoder, default_colbert_repo_path
    from materialize.db_cacheables import iter_db_records
    from run_nq_materialized_dense_regions import _region_cacheable
    from subchunk_store import cacheables_from_metadata

    folder = Path(cfg["artifacts"])
    if (folder / "regions_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    encoder = ColBERTEncoder(
        model_name=cfg["env"]["COLBERT_MODEL_NAME"],
        device="cpu",
        batch_size=1,
        repo_path=default_colbert_repo_path(),
    )
    override = RegionOverride(
        encoder, cfg["env"]["DB_DIR"], cfg["window"], rebuild=True
    )
    artifact = ColBERTWindowArtifact(cfg["env"]["COLBERT_WINDOW_DIR"])
    target = folder / "regions.sqlite3"
    staging = folder / "regions.building.sqlite3"
    staging.unlink(missing_ok=True)
    connection = sqlite3.connect(staging)
    connection.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_ids TEXT, specs TEXT)"
    )
    connection.execute(
        "CREATE TABLE regions (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, text TEXT)"
    )
    chunks = total = 0
    digest = hashlib.sha256()
    for cid, metadata in iter_db_records(cfg["env"]["DB_DIR"]):
        units = cacheables_from_metadata(metadata, override.subchunks)
        doc = RetrievableChunk(id=cid, text="", cacheables=units, metadata=metadata)
        specs = override.specs(doc, units, artifact)
        source_ids = [str(c.id) for c in units if c.text]
        connection.execute(
            "INSERT INTO chunks VALUES (?,?,?)",
            (str(cid), json.dumps(source_ids), json.dumps(specs)),
        )
        for center, indices in specs:
            region = _region_cacheable(
                doc=doc, center_idx=center, selected_indices=indices
            )
            connection.execute(
                "INSERT INTO regions VALUES (?,?,?)",
                (total, str(region.id), region.text),
            )
            digest.update(
                json.dumps([cid, center, indices, region.text]).encode() + b"\n"
            )
            total += 1
        chunks += 1
    connection.commit()
    connection.close()
    staging.replace(target)
    save(
        folder / "regions_complete.json",
        dict(
            tokenizer=encoder.model_name,
            window=cfg["window"],
            token_overhead=encoder.doc_token_overhead,
            chunks=chunks,
            regions=total,
            memberships_and_text_sha256=digest.hexdigest(),
            scope="all chunks of the existing DB; same RegionOverride as completed LateOn sweep",
        ),
    )
    print("Shared regions prepared", chunks, total, flush=True)


def generate(cfg):
    import numpy as np
    import torch
    from compressor import factory
    from compressor.methods.dense import DenseSlidingSubchunkCompressor
    from run_hotpot_lighton_suite import generate as main_generate

    folder = Path(cfg["artifacts"])
    info = read(folder / "regions_complete.json")
    complete = read(folder / "dense_complete.json")
    assert info["window"] == cfg["window"]
    assert complete["compute_dtype"] == "float32"
    connection = sqlite3.connect(
        (folder / "regions.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    vectors = np.load(folder / "dense_vectors.npy", mmap_mode="r")
    assert vectors.shape == (info["regions"], 768)
    assert vectors.dtype == np.float16 and np.isfinite(vectors).all()

    class SharedRegionDense(DenseSlidingSubchunkCompressor):
        def __init__(self):
            super().__init__()
            self.region_token_budget = cfg["window"]
            for row, rid, text in connection.execute(
                "SELECT rowid,id,text FROM regions"
            ):
                self._runtime_cacheable_embeddings[rid] = (
                    text,
                    torch.tensor(vectors[row].copy(), dtype=torch.float32),
                )

        def _cached_sliding_region_specs(self, doc, cacheables):
            row = connection.execute(
                "SELECT source_ids,specs FROM chunks WHERE id=?", (str(doc.id),)
            ).fetchone()
            assert row is not None, doc.id
            assert json.loads(row[0]) == [str(c.id) for c in cacheables]
            return [(center, tuple(indices)) for center, indices in json.loads(row[1])]

        def _embed_unique_cacheables(self, cacheables):
            for unit in cacheables:
                if str(unit.id) not in self._runtime_cacheable_embeddings:
                    raise KeyError(f"Unmaterialized region: {unit.id}")
            return super()._embed_unique_cacheables(cacheables)

    factory.COMPRESSOR_TYPES["dense_sliding_subchunk"] = SharedRegionDense
    main_generate(cfg, cfg["method"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["prepare", "dense", "generate", "run"], required=True
    )
    args = parser.parse_args()
    cfg = read(args.config)
    os.environ.update(cfg["env"])
    from build_full_lighton_artifacts import initialize, sha, save

    for path, expected in cfg["input_hashes"].items():
        assert sha(Path(path)) == expected, path
    initialize()
    if args.action == "prepare":
        prepare(cfg)
    elif args.action == "dense":
        from build_native180_regions import dense

        dense(cfg)
    elif args.action == "generate":
        generate(cfg)
    else:
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
        save(Path(cfg["output"]) / "complete.json", read(folder / "results.json"))


if __name__ == "__main__":
    main()
