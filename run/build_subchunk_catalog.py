"""Rebuild sentence/region metadata in an offline copy of an existing retrieval DB."""

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chunk import CacheableChunk
from encoder.lateon import LATEON_MODEL, LATEON_REVISION
from materialize.colbert_materializer import (
    ColBERTWindowEncoder,
    _window_bounded_region_index_specs,
)
from materialize.db_manifest import write_db_build_manifest
from materialize.splitter.base import SentenceWiseSplitter
from materialize.splitter.parser.sentence import SentenceParser
from subchunk_store import SubchunkWriter, pack_regions, unpack_regions
from transformers import AutoTokenizer

GROUPS = None
SPLITTER = None
NATIVE = None
WINDOW = None
MAX_UNIT = None
LEGACY_CHUNKS = None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_path(args):
    return args.source_manifest or args.source_db / "build_manifest.json"


def read_groups(db_dir, legacy_chunks_path=None):
    con = sqlite3.connect((db_dir / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True)
    groups = defaultdict(list)
    legacy = None
    if legacy_chunks_path is not None:
        values = json.loads(legacy_chunks_path.read_text())
        legacy = {row["id"]: row for row in values}
        assert len(legacy) == len(values)
    query = """SELECT e.id,e.embedding_id,p.string_value,p.int_value,s.int_value,t.int_value,
        a.int_value,d.string_value
        FROM embeddings e
        JOIN segments seg ON seg.id=e.segment_id
        JOIN collections c ON c.id=seg.collection AND c.name='doc_collection'
        JOIN embedding_metadata p ON p.id=e.id AND p.key='parent_doc_id'
        JOIN embedding_metadata s ON s.id=e.id AND s.key='source_token_start'
        JOIN embedding_metadata t ON t.id=e.id AND t.key='source_token_end'
        LEFT JOIN embedding_metadata a ON a.id=e.id AND a.key='subchunk_start'
        JOIN embedding_metadata d ON d.id=e.id AND d.key='chroma:document'
        ORDER BY e.id"""
    for (
        sql_id,
        cid,
        parent,
        parent_integer,
        start,
        end,
        old_unit_start,
        text,
    ) in con.execute(query):
        if legacy is not None:
            stored = legacy[cid]
            assert stored["document_id"] == parent_integer
            assert stored["metadata"]["parent_doc_id"] == parent_integer
            assert (
                stored["metadata"]["source_token_start"],
                stored["metadata"]["source_token_end"],
            ) == (start, end)
            assert stored["text"] == text
            parent = f"{stored['document_id']}.txt"
            old_unit_start = stored["subchunk_start"]
        assert isinstance(parent, str) and type(old_unit_start) is int
        groups[parent].append(
            (
                sql_id,
                cid,
                start,
                end,
                old_unit_start,
                hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    total = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert sum(map(len, groups.values())) == total
    if legacy is not None:
        assert {row[1] for rows in groups.values() for row in rows} == set(legacy)
    collection = con.execute(
        "SELECT id,schema_str FROM collections WHERE name='doc_collection'"
    ).fetchone()
    assert collection
    columns = [x[1] for x in con.execute("PRAGMA table_info(embedding_metadata_array)")]
    assert columns == [
        "id",
        "key",
        "string_value",
        "int_value",
        "float_value",
        "bool_value",
    ], columns
    con.close()
    return dict(groups), total, collection


def document(parent):
    result = SPLITTER.split_document(parent)
    expected = {row[1]: row for row in GROUPS[parent]}
    expected_ranges = {(row[2], row[3]): row for row in GROUPS[parent]}
    assert len(expected_ranges) == len(expected)
    assert len(result.retrievable_chunks) == len(expected), parent
    texts = [unit.text for unit in result.chunks]
    source = (Path(SPLITTER.docs_dir) / parent).read_text()
    assert "".join(source.split()) == "".join(
        "".join(text.split()) for text in texts
    ), parent
    native_counts = [
        len(ids)
        for ids in NATIVE(
            texts, add_special_tokens=False, truncation=False, verbose=False
        )["input_ids"]
    ]
    assert all(0 < count <= MAX_UNIT for count in native_counts), (
        parent,
        max(native_counts),
    )
    builder = SimpleNamespace(
        token_counts_without_specials=lambda value: native_counts,
        doc_token_overhead=len(NATIVE("", add_special_tokens=True)["input_ids"]) + 1,
        _next_window_candidate=ColBERTWindowEncoder._next_window_candidate,
    )
    windows = ColBERTWindowEncoder.build_centered_windows(builder, texts, WINDOW)
    maps = {
        str(i): [str(j) for j in spec.selected_indices]
        for i, spec in enumerate(windows)
    }
    positions = {unit.id: i for i, unit in enumerate(result.chunks)}
    rows = []
    covered = set()
    for chunk in result.retrievable_chunks:
        if LEGACY_CHUNKS is not None:
            original = expected_ranges[
                (
                    chunk.metadata["source_token_start"],
                    chunk.metadata["source_token_end"],
                )
            ]
        else:
            original = expected[chunk.id]
        assert chunk.metadata["source_token_start"] == original[2], chunk.id
        assert chunk.metadata["source_token_end"] == original[3], chunk.id
        assert hashlib.sha256(chunk.text.encode()).hexdigest() == original[5], chunk.id
        indices = [positions[unit.id] for unit in chunk.cacheables]
        assert indices == list(range(indices[0], indices[-1] + 1)), chunk.id
        covered.update(indices)
        regions = _window_bounded_region_index_specs([str(i) for i in indices], maps)
        packed = pack_regions(regions)
        rows.append(
            (
                original[0],
                original[1],
                indices[0],
                indices[-1] + 1,
                packed["region_starts"],
                packed["region_ends"],
            )
        )
    assert covered == set(range(len(texts))), parent
    return {
        "parent": parent,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "units": [(unit.text, unit.prompt_token_count) for unit in result.chunks],
        "rows": rows,
        "max_native_tokens": max(native_counts),
    }


def prepare(args, groups):
    global GROUPS, SPLITTER, NATIVE, WINDOW, MAX_UNIT, LEGACY_CHUNKS
    marker = args.log_dir / "preparation_complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    GROUPS, WINDOW, MAX_UNIT = groups, args.region_tokens, args.max_unit_tokens
    LEGACY_CHUNKS = args.legacy_chunks
    source_manifest = json.loads(source_manifest_path(args).read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        source_manifest["tokenizer_name"], local_files_only=True
    )
    tokenizer.model_max_length = 10**9
    NATIVE = AutoTokenizer.from_pretrained(
        LATEON_MODEL, revision=LATEON_REVISION, local_files_only=True
    )
    NATIVE.model_max_length = 10**9
    SPLITTER = SentenceWiseSplitter(
        str(args.docs_dir),
        tokenizer,
        None,
        source_manifest["retrievable_chunk_size"],
        args.max_unit_tokens,
        parser=SentenceParser(tokenizer, long_unit_tokenizer=NATIVE),
    )
    for filename in ["subchunks.json", "subchunks.json.tmp"]:
        (args.output_db / filename).unlink(missing_ok=True)
    update_path = args.log_dir / "updates.sqlite3"
    update_path.unlink(missing_ok=True)
    updates = sqlite3.connect(update_path)
    updates.execute(
        "CREATE TABLE updates (sql_id INTEGER PRIMARY KEY,cid TEXT UNIQUE,a INTEGER,b INTEGER,starts TEXT,ends TEXT)"
    )
    writer = SubchunkWriter(args.output_db)
    parents = sorted(groups, key=lambda parent: min(row[4] for row in groups[parent]))
    assert {p.name for p in args.docs_dir.glob("*.txt")} == set(parents)
    count = regions = maximum = 0
    begun = time.monotonic()
    try:
        with (
            mp.get_context("fork").Pool(args.workers) as pool,
            (args.log_dir / "documents.jsonl").open("w") as audit,
        ):
            for done, value in enumerate(pool.imap(document, parents, chunksize=8), 1):
                start = writer.count
                cacheables = [
                    CacheableChunk(id=i, text=text, prompt_token_count=tokens)
                    for i, (text, tokens) in enumerate(value["units"])
                ]
                writer.add_document(cacheables)
                for sql_id, cid, a, b, starts, ends in value["rows"]:
                    updates.execute(
                        "INSERT INTO updates VALUES (?,?,?,?,?,?)",
                        (
                            sql_id,
                            cid,
                            start + a,
                            start + b,
                            json.dumps(starts),
                            json.dumps(ends),
                        ),
                    )
                    count += 1
                    regions += len(starts)
                maximum = max(maximum, value["max_native_tokens"])
                audit.write(
                    json.dumps(
                        {
                            "document": value["parent"],
                            "source_sha256": value["source_sha256"],
                            "unit_start": start,
                            "unit_end": writer.count,
                            "chunks": len(value["rows"]),
                            "max_native_tokens": value["max_native_tokens"],
                        }
                    )
                    + "\n"
                )
                if done % 500 == 0 or done == len(parents):
                    updates.commit()
                    audit.flush()
                    state = dict(
                        stage="sentences_and_regions",
                        documents=done,
                        total_documents=len(parents),
                        chunks=count,
                        sentences=writer.count,
                        regions=regions,
                        seconds=round(time.monotonic() - begun, 1),
                    )
                    save(args.log_dir / "progress.json", state)
                    print(json.dumps(state), flush=True)
        updates.commit()
        writer.finish()
    finally:
        writer.close()
        updates.close()
    result = dict(
        documents=len(parents),
        sentences=writer.count,
        chunks=count,
        regions=regions,
        max_native_tokens=maximum,
        seconds=time.monotonic() - begun,
    )
    save(marker, result)
    return result


def clone(args, collection_id):
    marker = args.log_dir / "clone_complete.json"
    if marker.exists():
        return
    assert not args.output_db.exists(), args.output_db
    args.output_db.mkdir(parents=True)
    begun = time.monotonic()
    source = sqlite3.connect(
        (args.source_db / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    target = sqlite3.connect(args.output_db / "chroma.sqlite3")
    last = [0.0]

    def progress(status, remaining, total):
        now = time.monotonic()
        if now - last[0] >= 10 or remaining == 0:
            state = dict(
                stage="copy_retrieval_sqlite",
                pages=total - remaining,
                total_pages=total,
                seconds=round(now - begun, 1),
            )
            save(args.log_dir / "progress.json", state)
            print(json.dumps(state), flush=True)
            last[0] = now

    source.backup(target, pages=4096, progress=progress)
    target.execute(
        "INSERT INTO collection_metadata (collection_id,key,bool_value) VALUES (?,'regions_ready',0) ON CONFLICT(collection_id,key) DO UPDATE SET bool_value=0",
        (collection_id,),
    )
    target.commit()
    target.close()
    source.close()
    indexes = {}
    for path in args.source_db.iterdir():
        if path.is_dir():
            shutil.copytree(path, args.output_db / path.name)
            for source_file in path.rglob("*"):
                if source_file.is_file():
                    relative = source_file.relative_to(args.source_db)
                    expected = sha256(source_file)
                    assert sha256(args.output_db / relative) == expected
                    indexes[str(relative)] = expected
    save(
        marker,
        dict(
            source_db=str(args.source_db),
            output_db=str(args.output_db),
            index_file_sha256=indexes,
            seconds=time.monotonic() - begun,
        ),
    )


def legacy_metadata_schema(schema):
    value = json.loads(schema)
    for key in ["subchunk_start", "subchunk_end", "region_starts", "region_ends"]:
        value["keys"].setdefault(key, {})["int"] = value["defaults"]["int"]
    value["keys"]["parent_doc_id"]["string"] = value["defaults"]["string"]
    return json.dumps(value, separators=(",", ":"))


def apply(args, collection_id, summary, groups, schema):
    marker = args.log_dir / "apply_complete.json"
    if marker.exists():
        return
    target = sqlite3.connect(args.output_db / "chroma.sqlite3")
    updates = sqlite3.connect(
        (args.log_dir / "updates.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    begun = time.monotonic()
    if args.legacy_chunks is not None:
        parent_rows = [
            (parent, row[0]) for parent, rows in groups.items() for row in rows
        ]
        target.executemany(
            "UPDATE embedding_metadata SET string_value=?,int_value=NULL WHERE id=? AND key='parent_doc_id'",
            parent_rows,
        )
        target.execute(
            "UPDATE collections SET schema_str=? WHERE id=?",
            (legacy_metadata_schema(schema), collection_id),
        )
        target.commit()
    print(json.dumps(dict(stage="replace_region_arrays")), flush=True)
    target.execute(
        "DELETE FROM embedding_metadata_array WHERE key IN ('region_starts','region_ends')"
    )
    target.commit()
    cursor = updates.execute(
        "SELECT sql_id,a,b,starts,ends FROM updates ORDER BY sql_id"
    )
    done = 0
    while rows := cursor.fetchmany(4096):
        scalars = []
        arrays = []
        for sql_id, a, b, starts, ends in rows:
            scalars.extend([(a, sql_id, "subchunk_start"), (b, sql_id, "subchunk_end")])
            arrays.extend((sql_id, "region_starts", v) for v in json.loads(starts))
            arrays.extend((sql_id, "region_ends", v) for v in json.loads(ends))
        if args.legacy_chunks is not None:
            target.executemany(
                "INSERT INTO embedding_metadata (id,key,int_value) VALUES (?,?,?) ON CONFLICT(id,key) DO UPDATE SET int_value=excluded.int_value",
                [(sql_id, key, value) for value, sql_id, key in scalars],
            )
        else:
            target.executemany(
                "UPDATE embedding_metadata SET int_value=? WHERE id=? AND key=?",
                scalars,
            )
        target.executemany(
            "INSERT INTO embedding_metadata_array (id,key,int_value) VALUES (?,?,?)",
            arrays,
        )
        target.commit()
        done += len(rows)
        state = dict(
            stage="apply_sentence_region_metadata",
            chunks=done,
            total_chunks=summary["chunks"],
            seconds=round(time.monotonic() - begun, 1),
        )
        save(args.log_dir / "progress.json", state)
        print(json.dumps(state), flush=True)
    assert done == summary["chunks"]
    updates.close()
    target.close()
    save(marker, dict(chunks=done, seconds=time.monotonic() - begun))


def verify(args, summary, collection_id, schema):
    from subchunk_store import load_subchunks
    import chromadb

    begun = time.monotonic()
    collection = chromadb.PersistentClient(path=str(args.output_db)).get_collection(
        "doc_collection", embedding_function=None
    )
    assert collection.count() == summary["chunks"]
    updates = sqlite3.connect(
        (args.log_dir / "updates.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    expected = {
        cid: (a, b, json.loads(s), json.loads(e))
        for cid, a, b, s, e in updates.execute(
            "SELECT cid,a,b,starts,ends FROM updates"
        )
    }
    seen = set()
    for offset in range(0, summary["chunks"], 4096):
        result = collection.get(limit=4096, offset=offset, include=["metadatas"])
        for cid, meta in zip(result["ids"], result["metadatas"], strict=True):
            a, b, starts, ends = expected[cid]
            assert (
                meta["subchunk_start"],
                meta["subchunk_end"],
                meta["region_starts"],
                meta["region_ends"],
            ) == (a, b, starts, ends), cid
            unpack_regions(meta, list(range(a, b)))
            assert cid not in seen
            seen.add(cid)
        if offset % 40960 == 0:
            state = dict(
                stage="verify_public_chroma_metadata",
                chunks=len(seen),
                total_chunks=summary["chunks"],
            )
            save(args.log_dir / "progress.json", state)
            print(json.dumps(state), flush=True)
    assert seen == set(expected)
    units = load_subchunks(str(args.output_db))
    assert len(units) == summary["sentences"]
    del units
    before = json.loads((args.log_dir / "clone_complete.json").read_text())
    for relative, digest in before["index_file_sha256"].items():
        assert sha256(args.output_db / relative) == digest, relative
    new_groups, new_count, new_collection = read_groups(args.output_db)
    old_groups, old_count, old_collection = read_groups(
        args.source_db, args.legacy_chunks
    )
    assert new_count == old_count == summary["chunks"]
    assert old_collection == (collection_id, schema)
    assert new_collection == (
        collection_id,
        legacy_metadata_schema(schema) if args.legacy_chunks is not None else schema,
    )
    for parent, rows in old_groups.items():
        assert [(r[0], r[1], r[2], r[3], r[5]) for r in rows] == [
            (r[0], r[1], r[2], r[3], r[5]) for r in new_groups[parent]
        ], parent
    common = collection.metadata or {}
    collection.modify(
        metadata={
            **common,
            "regions_ready": True,
            "region_token_budget": args.region_tokens,
            "region_tokenizer_name": LATEON_MODEL,
            "region_tokenizer_revision": LATEON_REVISION,
            "max_subchunk_tokens": args.max_unit_tokens,
            "max_subchunk_tokenizer_name": LATEON_MODEL,
            "max_subchunk_tokenizer_revision": LATEON_REVISION,
        }
    )
    manifest = json.loads(source_manifest_path(args).read_text())
    if args.legacy_chunks is not None:
        manifest = {
            **manifest,
            "legacy_source_manifest": dict(manifest),
            "retrieval_db_created": True,
            "compatibility_note": "Actual Chroma DB with reused retrieval vectors/graph and rebuilt row-indexed sentence/region metadata.",
            "num_papers": summary["documents"],
            "num_cacheables": summary["sentences"],
            "span_note": "LateOn-bounded source-character pieces mapped to original retrieval-token positions.",
        }
    manifest.update(
        splitter="sentence",
        max_subchunk_tokens=args.max_unit_tokens,
        max_subchunk_tokenizer_name=LATEON_MODEL,
        max_subchunk_tokenizer_revision=LATEON_REVISION,
        region_token_budget=args.region_tokens,
        region_tokenizer_name=LATEON_MODEL,
        region_tokenizer_revision=LATEON_REVISION,
        sentence_boundary_rules=[
            "blingfire_conservative_merge",
            "native_offset_alignment",
            "literal_2plus_newlines",
            "http_https_isolation",
        ],
        source_db=str(args.source_db),
        source_build_manifest_sha256=sha256(source_manifest_path(args)),
        retrieval_vectors_reused=True,
        retrieval_graph_reused=True,
        sentence_parser_sha256=sha256(
            ROOT / "src/materialize/splitter/parser/sentence.py"
        ),
    )
    write_db_build_manifest(args.output_db, manifest)
    updates.close()
    final = dict(
        **summary,
        output_db=str(args.output_db),
        long_unit_tokenizer=LATEON_MODEL,
        long_unit_max_tokens=args.max_unit_tokens,
        region_tokenizer=LATEON_MODEL,
        region_tokens=args.region_tokens,
        public_api_verified_chunks=len(seen),
        retrieval_source_ranges_text_ids_and_schema_unchanged=args.legacy_chunks
        is None,
        retrieval_source_ranges_text_ids_unchanged=True,
        retrieval_vector_schema_unchanged=True,
        legacy_metadata_adapted=args.legacy_chunks is not None,
        retrieval_index_files_byte_identical=True,
        embeddings_generated=False,
        verification_seconds=time.monotonic() - begun,
    )
    save(args.log_dir / "complete.json", final)
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-unit-tokens", type=int, default=100)
    parser.add_argument("--region-tokens", type=int, default=180)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--legacy-chunks", type=Path)
    args = parser.parse_args()
    for name in ["source_db", "docs_dir", "output_db", "log_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    assert args.source_db != args.output_db
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if (args.log_dir / "complete.json").exists():
        print((args.log_dir / "complete.json").read_text())
        return
    save(
        args.log_dir / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_parser_sha256": sha256(
                ROOT / "src/materialize/splitter/parser/sentence.py"
            ),
        },
    )
    groups, total, (collection_id, schema) = read_groups(
        args.source_db, args.legacy_chunks
    )
    print(
        json.dumps(dict(stage="source_verified", documents=len(groups), chunks=total)),
        flush=True,
    )
    clone(args, collection_id)
    summary = prepare(args, groups)
    assert summary["chunks"] == total
    apply(args, collection_id, summary, groups, schema)
    verify(args, summary, collection_id, schema)


if __name__ == "__main__":
    main()
