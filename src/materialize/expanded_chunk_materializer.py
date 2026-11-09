"""Encode persisted boundary-complete chunk sentences with LateOn.

One vector row is one retrieval chunk, not a globally shared sentence. Sentence
positions address that chunk's vectors. No retrieval DB or sentence text changes.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoTokenizer

import colbert_artifact as ca
from encoder.colbert import ColBERTEncoder, default_colbert_repo_path
from encoder.lateon import LATEON_MODEL, LATEON_REVISION
from materialize.colbert_residual import (
    _sha256_file,
    compress_colbert_window_artifact,
)
from materialize.db_cacheables import iter_db_records
from subchunk_store import load_subchunks

FORMAT = "lateon_boundary_complete_chunks_v1"


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compose(start, end, subchunks):
    if not 0 <= start < end <= len(subchunks):
        raise ValueError("invalid persisted sentence range")
    parts, spans, position = [], [], 0
    for sid in range(start, end):
        text = subchunks[sid][0]
        if parts:
            position += 1
        spans.append((sid, position, position + len(text)))
        parts.append(text)
        position += len(text)
    return " ".join(parts), spans


def sentence_ranges(offsets, spans):
    """Positive character overlap; a crossing token may belong to both units."""
    result = []
    for sid, begin, end in spans:
        indices = np.flatnonzero((offsets[:, 1] > begin) & (offsets[:, 0] < end))
        if len(indices):
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise ValueError("noncontiguous sentence token positions")
            result.append((sid, begin, end, int(indices[0]), int(indices[-1]) + 1))
        else:
            result.append((sid, begin, end, 0, 0))
    return result


def prepare(db_dir, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: _sha256_file(Path(db_dir) / name)
        for name in ("build_manifest.json", "subchunks.json")
    }
    if (work / "prepared.json").exists():
        prepared = json.loads((work / "prepared.json").read_text())
        if prepared["source_hashes"] != source_hashes:
            raise ValueError("source changed since preparation")
        return prepared
    subchunks = load_subchunks(str(Path(db_dir).resolve()))
    tokenizer = AutoTokenizer.from_pretrained(
        LATEON_MODEL, revision=LATEON_REVISION, local_files_only=True
    )
    connection = sqlite3.connect(work / "inputs.sqlite3")
    connection.execute("drop table if exists chunks")
    connection.execute(
        "create table chunks(row integer primary key, id text unique, parent text, "
        "sentence_start integer, sentence_end integer, input_tokens integer)"
    )
    pending, lengths, row = [], [], 0

    def flush():
        nonlocal row
        texts = [
            compose(m["subchunk_start"], m["subchunk_end"], subchunks)[0]
            for _, m in pending
        ]
        ids = tokenizer(
            [t.strip() for t in texts], truncation=False, padding=False, verbose=False
        )["input_ids"]
        for (cid, meta), tokens in zip(pending, ids, strict=True):
            length = len(tokens) + 1
            if length > 8192:
                raise ValueError(
                    f"chunk {cid}: {length} tokens exceeds8192; no truncation"
                )
            connection.execute(
                "insert into chunks values(?,?,?,?,?,?)",
                (
                    row,
                    str(cid),
                    str(meta["parent_doc_id"]),
                    meta["subchunk_start"],
                    meta["subchunk_end"],
                    length,
                ),
            )
            lengths.append(length)
            row += 1
        connection.commit()
        pending.clear()
        print(json.dumps({"stage": "prepare", "chunks": row}), flush=True)

    for cid, meta in iter_db_records(db_dir, batch_size=2048):
        pending.append((cid, meta))
        if len(pending) == 2048:
            flush()
    if pending:
        flush()
    connection.close()
    if not lengths:
        raise ValueError("empty retrieval corpus")
    prepared = dict(
        db_dir=str(Path(db_dir).resolve()),
        source_hashes=source_hashes,
        chunks=row,
        input_tokens=sum(lengths),
        max_input_tokens=max(lengths),
        mean_input_tokens=float(np.mean(lengths)),
        over512=sum(n > 512 for n in lengths),
        model=LATEON_MODEL,
        revision=LATEON_REVISION,
        fp16_upper_bound_bytes=sum(lengths) * 256,
        compressed_upper_bound_bytes=sum(lengths) * 40,
        input="space-joined complete persisted sentences attached to each DB chunk",
    )
    save(work / "prepared.json", prepared)
    return prepared


def encode_batch(encoder, records, subchunks):
    composed = [compose(r[3], r[4], subchunks) for r in records]
    texts = [t for t, _ in composed]
    lengths = encoder.token_counts(texts)
    if lengths != [r[5] for r in records] or max(lengths) > encoder.doc_maxlen:
        raise ValueError("prepared length changed or input would be truncated")
    ids, attention, offsets = encoder.tensorize_docs_with_offsets(texts)
    if attention.sum(1).tolist() != lengths:
        raise ValueError("tokenizer truncated the input")
    with torch.inference_mode():
        vectors, valid = encoder.checkpoint.doc(
            ids, attention, keep_dims="return_mask", to_cpu=True
        )
    if not torch.isfinite(vectors).all():
        raise ValueError("nonfinite LateOn vectors")
    result = []
    for i, (_, spans) in enumerate(composed):
        off = offsets[i].cpu().numpy()
        live = valid[i].squeeze(-1).cpu().numpy().astype(bool) & (off[:, 1] > off[:, 0])
        off = off[live]
        values = (
            vectors[i, torch.from_numpy(live)].contiguous().to(torch.float16).numpy()
        )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite stored FP16 vectors")
        # Keep every source token, including separator-only tokens. No heuristic
        # reassignment to a sentence: offsets preserve the exact original mapping.
        result.append((values, off.astype(np.int32), sentence_ranges(off, spans)))
    return result


def tune_batch(encoder, record, subchunks, work):
    batch = 1
    best = 1
    measurements = []
    while batch <= 512:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = encode_batch(encoder, [record] * batch, subchunks)
            del result
            peak = torch.cuda.max_memory_allocated()
            measurements.append(
                dict(
                    batch=batch, seconds=time.perf_counter() - started, peak_bytes=peak
                )
            )
            best = batch
            if peak > torch.cuda.get_device_properties(0).total_memory * 0.80:
                break
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            break
    save(
        work / "batch_probe.json",
        dict(batch_size=best, longest_input=record[5], measurements=measurements),
    )
    return best


def build(db_dir, work):
    work = Path(work)
    prepared = prepare(db_dir, work)
    fp = work / "fp16"
    if (fp / "index.json").exists():
        return
    free = shutil.disk_usage(work).free
    required = (
        prepared["fp16_upper_bound_bytes"] + prepared["compressed_upper_bound_bytes"]
    )
    if free < required + 2 * 1024**3:
        raise RuntimeError(
            f"insufficient space: available={free}, required_upper_bound={required}"
        )
    (fp / "data").mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(work / "inputs.sqlite3")
    mapping = sqlite3.connect(fp / "chunks.sqlite3")
    mapping.execute(
        "create table if not exists chunks(row integer primary key, id text unique, parent text, sentence_start integer, sentence_end integer, token_start integer, token_end integer, input_tokens integer)"
    )
    mapping.execute(
        "create table if not exists sentences(chunk_row integer, sentence_id integer, char_start integer, char_end integer, token_start integer, token_end integer, primary key(chunk_row,sentence_id))"
    )
    completed, tokens = mapping.execute(
        "select count(*),coalesce(max(token_end),0) from chunks"
    ).fetchone()
    subchunks = load_subchunks(str(Path(db_dir).resolve()))
    torch.manual_seed(123)
    torch.set_num_threads(16)
    encoder = ColBERTEncoder(
        model_name=LATEON_MODEL,
        device="cuda",
        batch_size=1,
        doc_maxlen=max(4, prepared["max_input_tokens"]),
        query_maxlen=32,
        repo_path=default_colbert_repo_path(),
        verify_tensorization=False,
    )
    probe = work / "batch_probe.json"
    batch = (
        json.loads(probe.read_text())["batch_size"]
        if probe.exists()
        else tune_batch(
            encoder,
            connection.execute(
                "select * from chunks order by input_tokens desc limit 1"
            ).fetchone(),
            subchunks,
            work,
        )
    )
    started = time.perf_counter()
    start_completed = completed
    vec_path, off_path = (
        fp / "data/vectors.fp16.bin",
        fp / "token_char_offsets.int32.bin",
    )
    with (
        vec_path.open("r+b" if vec_path.exists() else "w+b") as vf,
        off_path.open("r+b" if off_path.exists() else "w+b") as of,
    ):
        for handle, size in ((vf, tokens * 256), (of, tokens * 8)):
            handle.truncate(size)
            handle.seek(size)
        while completed < prepared["chunks"]:
            records = connection.execute(
                "select * from chunks where row>=? order by row limit ?",
                (completed, batch),
            ).fetchall()
            try:
                encoded = encode_batch(encoder, records, subchunks)
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch //= 2
                gc.collect()
                torch.cuda.empty_cache()
                print(f"OOM retry batch={batch}", flush=True)
                continue
            for record, (values, offsets, spans) in zip(records, encoded, strict=True):
                row, cid, parent, begin, end, length = record
                if row != completed:
                    raise ValueError("nonconsecutive chunk row")
                vf.write(values.tobytes())
                of.write(offsets.tobytes())
                mapping.execute(
                    "insert into chunks values(?,?,?,?,?,?,?,?)",
                    (
                        row,
                        cid,
                        parent,
                        begin,
                        end,
                        tokens,
                        tokens + len(values),
                        length,
                    ),
                )
                mapping.executemany(
                    "insert into sentences values(?,?,?,?,?,?)",
                    [(row, *span) for span in spans],
                )
                tokens += len(values)
                completed += 1
            vf.flush()
            of.flush()
            os.fsync(vf.fileno())
            os.fsync(of.fileno())
            mapping.commit()
            progress = dict(
                stage="encoding",
                completed=completed,
                total=prepared["chunks"],
                tokens=tokens,
                batch_size=batch,
                elapsed_sec=time.perf_counter() - started,
                chunks_per_sec=(completed - start_completed)
                / max(time.perf_counter() - started, 1e-6),
            )
            save(work / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    offsets = np.array(
        [0]
        + [r[0] for r in mapping.execute("select token_end from chunks order by row")],
        dtype=np.int64,
    )
    np.save(fp / "data/offsets.npy", offsets)
    mapping.close()
    connection.close()
    save(
        fp / "data/index.json",
        dict(
            format=ca.DATA_ARTIFACT_FORMAT,
            embedding_storage="fp16",
            embedding_dim=128,
            num_tokens=tokens,
            num_cacheables=completed,
            subchunk_id_layout="vector_row",
            row_unit="expanded_chunk",
            region_token_budget=0,
            offsets_file="offsets.npy",
            vectors_file="vectors.fp16.bin",
        ),
    )
    _, reference = ca.build_db_manifest_reference(db_dir=db_dir, artifact_dir=fp)
    save(
        fp / "index.json",
        dict(
            format=ca.ARTIFACT_FORMAT,
            artifact_variant=FORMAT,
            row_unit="expanded_chunk",
            db_manifest=reference,
            model_name=LATEON_MODEL,
            model_revision=LATEON_REVISION,
            official_doc_maxlen=encoder.doc_maxlen,
            embedding_dim=128,
            scope="all_retrieval_chunks",
            sentence_mapping="chunks.sqlite3",
            token_char_offsets="token_char_offsets.int32.bin",
            preparation=prepared,
            mask_punctuation=encoder.checkpoint.colbert_config.mask_punctuation,
            input_composition="space-joined persisted complete sentence units; no added title or prefix text",
            sentence_token_membership="positive character overlap, end exclusive; separator tokens retained in chunk",
            seed=123,
        ),
    )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


class ExpandedChunkArtifact:
    """Read chunk vectors and exact chunk-conditioned sentence slices."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "index.json").read_text())
        if self.index.get("format") != FORMAT:
            raise ValueError("not a finalized boundary-complete chunk artifact")
        self.data = ca.ColBERTWindowData(self.path / "data")
        self.mapping = sqlite3.connect(
            f"file:{self.path / 'chunks.sqlite3'}?mode=ro", uri=True
        )

    def vectors_for_chunk(self, chunk_id):
        found = self.mapping.execute(
            "select row from chunks where id=?", (str(chunk_id),)
        ).fetchone()
        if found is None:
            raise KeyError(chunk_id)
        return self.data.vectors_for_cacheable_ids([found[0]])[0]

    def vectors_for_doc(self, doc):
        found = self.mapping.execute(
            "select row,sentence_start,sentence_end from chunks where id=?",
            (str(doc.id),),
        ).fetchone()
        if found is None:
            raise KeyError(doc.id)
        row, begin, end = found
        if [c.id for c in doc.cacheables] != list(range(begin, end)):
            raise ValueError("runtime sentence membership differs from encoded chunk")
        vectors = self.data.vectors_for_cacheable_ids([row])[0]
        return [
            vectors[a:b]
            for a, b in self.mapping.execute(
                "select token_start,token_end from sentences where chunk_row=? order by sentence_id",
                (row,),
            )
        ]


def finalize(work, output):
    work, output = Path(work), Path(output)
    fp = work / "fp16"
    pending = output.with_name("." + output.name + ".compressed")
    if output.exists():
        if not (output / "validation.json").exists():
            raise ValueError("existing artifact lacks completion validation")
        return
    if not pending.exists():
        compress_colbert_window_artifact(fp, pending, seed=123, use_gpu=True)
    for name in ("chunks.sqlite3", "token_char_offsets.int32.bin"):
        shutil.copy2(fp / name, pending / name)
    index = json.loads((pending / "index.json").read_text())
    index["format"] = FORMAT
    save(pending / "index.json", index)
    check = ExpandedChunkArtifact(pending)
    source = ca.ColBERTWindowData(fp / "data")
    if not np.array_equal(check.data.offsets, source.offsets):
        raise ValueError("compression changed chunk token boundaries")
    expected = index["preparation"]["chunks"]
    if check.mapping.execute("select count(*) from chunks").fetchone()[0] != expected:
        raise ValueError("incomplete chunk coverage")
    for row, start, end, token_start, token_end in check.mapping.execute(
        "select row,sentence_start,sentence_end,token_start,token_end from chunks order by row"
    ):
        if (token_start, token_end) != tuple(check.data.offsets[row : row + 2]):
            raise ValueError("chunk mapping and vector offsets differ")
        spans = check.mapping.execute(
            "select sentence_id,token_start,token_end from sentences where chunk_row=? order by sentence_id",
            (row,),
        ).fetchall()
        if [s[0] for s in spans] != list(range(start, end)) or any(
            not 0 <= a <= b <= token_end - token_start for _, a, b in spans
        ):
            raise ValueError("invalid sentence mapping")
    if (
        pending / "token_char_offsets.int32.bin"
    ).stat().st_size != check.data.num_tokens * 8:
        raise ValueError("incomplete token offsets")
    # Validate all compressed record codes/norms in bounded reads, then decode
    # deterministic rows to exercise the normal runtime decompressor.
    for first in range(0, check.data.num_tokens, 1_000_000):
        records = check.data.token_records[first : first + 1_000_000]
        if (
            (records["code"] < 0).any()
            or (records["code"] >= len(check.data.centroids)).any()
            or not np.isfinite(records["norm"]).all()
            or (records["norm"] <= 0).any()
        ):
            raise ValueError("invalid compressed record")
    for row in sorted(set(np.linspace(0, expected - 1, min(32, expected), dtype=int))):
        vectors = check.data.vectors_for_cacheable_ids([int(row)])[0]
        if not torch.isfinite(vectors).all():
            raise ValueError("nonfinite reconstructed vectors")
    check.mapping.close()
    save(
        pending / "validation.json",
        dict(
            chunks=expected,
            num_tokens=check.data.num_tokens,
            all_chunk_and_sentence_ranges_checked=True,
            all_compressed_codes_norms_checked=True,
            fp16_vectors_sha256=check.data.index["source_vectors_sha256"],
            completed_at=time.time(),
        ),
    )
    pending.rename(output)
    del source, check
    gc.collect()
    shutil.rmtree(fp)
    save(
        work / "progress.json",
        dict(stage="complete", output=str(output), chunks=expected),
    )
