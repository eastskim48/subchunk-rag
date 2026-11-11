"""Read cacheable subchunks persisted in Chroma retrieval metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from chunk import CacheableChunk
from subchunk_store import load_subchunks, cacheables_from_metadata


def _cacheable_source_order(cacheable: CacheableChunk) -> int:
    if type(cacheable.id) is not int or cacheable.id < 0:
        raise ValueError(f"expected integer subchunk ID, got {cacheable.id!r}")
    return cacheable.id


def iter_db_records(db_dir: str | Path, batch_size: int = 2048):
    """Read Chroma metadata through its array-preserving public interface."""
    import chromadb

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(
        "doc_collection", embedding_function=None
    )
    for offset in range(0, collection.count(), batch_size):
        batch = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        yield from zip(batch["ids"], batch["metadatas"])


def iter_db_cacheables(db_dir: str | Path, batch_size: int = 2048):
    for _, cacheables in iter_db_cacheable_groups(db_dir, batch_size):
        yield from cacheables


def iter_unique_db_cacheables_by_document(
    db_dir: str | Path, batch_size: int = 2048
) -> Iterable[tuple[str, list[CacheableChunk]]]:
    """Stream deduplicated source-ordered cacheables one parent document at a time."""

    current_doc_id: str | None = None
    current_by_id: dict[str, CacheableChunk] = {}
    completed_doc_ids: set[str] = set()

    def finish_current():
        if current_doc_id is None:
            return None
        ordered = sorted(current_by_id.values(), key=_cacheable_source_order)
        return current_doc_id, [cacheable for cacheable in ordered if cacheable.text]

    for cacheable in iter_db_cacheables(db_dir=db_dir, batch_size=batch_size):
        cacheable_id = str(cacheable.id)
        if not cacheable_id:
            raise ValueError("DB cacheable requires a non-empty stable id")
        parent_doc_id = cacheable.parent_doc_id
        if not isinstance(parent_doc_id, str) or not parent_doc_id:
            raise ValueError(
                "DB cacheable requires parent_doc_id for subchunk materialization: "
                f"{cacheable.id}"
            )

        if current_doc_id is None:
            if parent_doc_id in completed_doc_ids:
                raise ValueError(
                    "DB cacheables for one parent document are not contiguous: "
                    f"{parent_doc_id}"
                )
            current_doc_id = parent_doc_id
        elif parent_doc_id != current_doc_id:
            finished = finish_current()
            if finished is not None:
                yield finished
            completed_doc_ids.add(current_doc_id)
            if parent_doc_id in completed_doc_ids:
                raise ValueError(
                    "DB cacheables for one parent document are not contiguous: "
                    f"{parent_doc_id}"
                )
            current_doc_id = parent_doc_id
            current_by_id = {}

        existing = current_by_id.get(cacheable_id)
        if existing is not None:
            if existing.to_payload() != cacheable.to_payload():
                raise ValueError(
                    "DB has conflicting payloads for duplicate cacheable id: "
                    f"{cacheable_id}"
                )
            continue
        current_by_id[cacheable_id] = cacheable

    finished = finish_current()
    if finished is not None:
        yield finished


def iter_db_cacheable_groups(db_dir: str | Path, batch_size: int = 2048):
    """Yield retrieval chunk IDs and subchunks from the shared row-indexed list."""
    subchunks = load_subchunks(str(Path(db_dir).resolve()))
    for chunk_id, metadata in iter_db_records(db_dir, batch_size):
        yield str(chunk_id), cacheables_from_metadata(metadata, subchunks)
