"""Build fixed BGE-token vanilla DBs while preserving original source text."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--parent-size", type=int)
    parser.add_argument("--namespace", default="vanilla-bge-small-v1.5-bgetok")
    args = parser.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CHROMA_EMBED_BACKEND="bge_small_v1_5",
        CHROMA_EMBED_DEVICE="cpu",
        CHROMA_EMBED_BATCH_SIZE="64",
    )
    import numpy as np
    import torch
    from transformers import AutoTokenizer, set_seed

    from chunk import CacheableChunk, RetrievableChunk
    from materialize.db_manifest import build_db_manifest, write_db_build_manifest
    from prompt import PromptProcessor
    from subchunk_store import SubchunkWriter
    from vectordb import ChromaDB

    torch.set_num_threads(8)
    set_seed(123)
    model = "BAAI/bge-small-en-v1.5"
    prompt_model = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        prompt_model, local_files_only=True
    )
    prompt = PromptProcessor(prompt_tokenizer, "")
    files = sorted(args.documents.glob("*.txt"))
    assert files
    prepared = {size: [] for size in args.sizes}
    sources = []
    for path in files:
        text = path.read_text()
        sources.append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        offsets = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )["offset_mapping"]
        assert offsets, path
        for size, rows in prepared.items():
            assert 0 < size <= (512 if args.parent_size else 510)
            if args.parent_size:
                assert args.parent_size % size == 0
            pieces = []
            for start in range(0, len(offsets), size):
                end = min(start + size, len(offsets))
                char_start = offsets[start][0] if start else 0
                char_end = offsets[end][0] if end < len(offsets) else len(text)
                piece = text[char_start:char_end]
                pieces.append(piece)
                # Retokenization at a WordPiece boundary can change token IDs.
                # Check the actual encoder input, without silently truncating it.
                input_length = len(tokenizer.encode(piece, truncation=False))
                if args.parent_size is None:
                    assert input_length <= 512, (path, start, input_length)
                count = len(
                    prompt_tokenizer.encode(
                        prompt.format_passage_chunk(piece), add_special_tokens=False
                    )
                )
                rows.append(
                    {
                        "id": f"{path.name}::token_{start}_{end}",
                        "parent_doc_id": path.name,
                        "source_token_start": start,
                        "source_token_end": end,
                        "source_char_start": char_start,
                        "source_char_end": char_end,
                        "text": piece,
                        "prompt_token_count": count,
                        "encoder_input_tokens": input_length,
                    }
                )
            assert "".join(pieces) == text, path
    save(args.output / "sources.json", sources)
    for size, rows in prepared.items():
        started = time.monotonic()
        db = args.dataset_root / f"{args.namespace}-{size}" / "db"
        if db.exists():
            raise FileExistsError(db)
        db.mkdir(parents=True)
        writer = SubchunkWriter(db)
        records = []
        try:
            for row, item in enumerate(rows):
                unit = CacheableChunk(
                    id=row,
                    text=item["text"],
                    parent_doc_id=item["parent_doc_id"],
                    prompt_token_count=item["prompt_token_count"],
                )
                writer.add_document([unit])
                records.append(
                    RetrievableChunk(
                        id=item["id"],
                        text=item["text"],
                        cacheables=[unit],
                        metadata={
                            key: item[key]
                            for key in (
                                "parent_doc_id",
                                "source_token_start",
                                "source_token_end",
                                "source_char_start",
                                "source_char_end",
                            )
                        },
                    )
                )
            writer.finish()
        finally:
            writer.close()
        vector_db = ChromaDB.for_build(
            str(db), embedding_device="cpu", embedding_batch_size=64
        )
        for start in range(0, len(records), 5461):
            batch = records[start : start + 5461]
            vector_db.store(batch)
            state = {
                "size": size,
                "encoded": start + len(batch),
                "total": len(records),
                "seconds": time.monotonic() - started,
            }
            save(args.output / "status.json", state)
            print(json.dumps(state), flush=True)
        assert vector_db.db.count() == len(records)
        for start in range(0, len(records), 512):
            batch = records[start : start + 512]
            stored = vector_db.db.get(
                ids=[r.id for r in batch],
                include=["embeddings", "documents", "metadatas"],
            )
            embeddings = np.asarray(stored["embeddings"])
            assert embeddings.shape == (len(batch), 384)
            assert np.isfinite(embeddings).all()
            assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
            expected = {r.id: r for r in batch}
            for cid, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            ):
                record = expected[cid]
                assert text == record.text
                assert all(meta[k] == v for k, v in record.metadata.items())
                assert meta["subchunk_start"] == record.cacheables[0].id
                assert meta["subchunk_end"] == record.cacheables[0].id + 1
        manifest = build_db_manifest(
            splitter="fixed_size",
            cacheable_chunk_size=size,
            retrievable_chunk_size=size,
            max_subchunk_tokens=None,
            tokenizer_name=model,
            deduplicate_documents_by_hash=False,
            embedding_backend="bge_small_v1_5",
            db_batch_size=5461,
            embedding_device="cpu",
            embedding_batch_size=64,
        )
        manifest.update(
            content_chunk_size=size,
            visible_suffix_tokens=0,
            split_special_tokens=False,
            split_overlap_tokens=0,
            text_construction="contiguous original-character slices at BGE token starts",
            prompt_tokenizer_name=prompt_model,
            prompt_counts="src.prompt.PromptProcessor.format_passage_chunk",
            seed=123,
            torch_threads=8,
            embedding_dtype="float32",
            nested_parent_size=args.parent_size,
            encoder_input_limit=512,
            encoder_truncation="unchanged native DenseTextEmbedder truncation=True",
            inputs_over_encoder_limit=sum(
                r["encoder_input_tokens"] > 512 for r in rows
            ),
            documents=len(files),
            entries=len(records),
            max_actual_encoder_input_tokens=max(
                r["encoder_input_tokens"] for r in rows
            ),
            hnsw_configuration=vector_db.db.configuration,
            scope="custom full COVID-QA document retrieval corpus; unchanged source documents",
        )
        write_db_build_manifest(db, manifest)
        save(
            args.output / f"complete_{size}.json",
            {"db": str(db), "seconds": time.monotonic() - started, **manifest},
        )
        print(f"Completed and validated {size}: {len(records)} entries", flush=True)


if __name__ == "__main__":
    main()
