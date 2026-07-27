"""Read cacheable subchunks persisted in Chroma retrieval metadata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from chunk import CacheableChunk


def _cacheable_source_order(cacheable: CacheableChunk) -> tuple[bool, int, str]:
    """Sort source-aligned cacheables deterministically within a document."""

    chunk_start = cacheable.chunk_start
    return (
        chunk_start is None,
        int(chunk_start) if chunk_start is not None else 0,
        str(cacheable.id),
    )


def iter_db_cacheables(
    db_dir: str | Path, batch_size: int = 2048
) -> Iterable[CacheableChunk]:
    """Yield cacheables in the retrieval-record order stored by Chroma."""

    sqlite_path = Path(db_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"missing Chroma sqlite database: {sqlite_path}")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        total_records = connection.execute(
            "select count(*) from embedding_metadata where key='cacheables_json'"
        ).fetchone()[0]
        query = (
            "select cacheables.string_value "
            "from embeddings e "
            "join embedding_metadata cacheables "
            "  on e.id = cacheables.id and cacheables.key = 'cacheables_json' "
            "order by e.id "
            "limit ? offset ?"
        )
        for offset in range(0, total_records, batch_size):
            rows = connection.execute(query, (batch_size, offset)).fetchall()
            for (cacheables_json,) in rows:
                payload = json.loads(cacheables_json) if cacheables_json else []
                for item in payload:
                    if isinstance(item, dict):
                        yield CacheableChunk.from_payload(item)
    finally:
        connection.close()


def group_unique_db_cacheables(
    cacheables: Iterable[CacheableChunk],
) -> tuple[dict[str, list[CacheableChunk]], dict[str, int]]:
    """Deduplicate overlapping DB cacheables and group them by source document."""

    unique_by_id: dict[str, CacheableChunk] = {}
    total_occurrences = 0
    duplicate_occurrences = 0
    empty_cacheables = 0

    for cacheable in cacheables:
        total_occurrences += 1
        cacheable_id = str(cacheable.id)
        if not cacheable_id:
            raise ValueError("DB cacheable requires a non-empty stable id")

        existing = unique_by_id.get(cacheable_id)
        if existing is not None:
            duplicate_occurrences += 1
            if existing.to_payload() != cacheable.to_payload():
                raise ValueError(
                    "DB has conflicting payloads for duplicate cacheable id: "
                    f"{cacheable_id}"
                )
            continue
        unique_by_id[cacheable_id] = cacheable

    grouped: dict[str, list[CacheableChunk]] = {}
    for cacheable in unique_by_id.values():
        if not cacheable.text:
            empty_cacheables += 1
            continue
        parent_doc_id = cacheable.parent_doc_id
        if not isinstance(parent_doc_id, str) or not parent_doc_id:
            raise ValueError(
                "DB cacheable requires parent_doc_id for candidate materialization: "
                f"{cacheable.id}"
            )
        grouped.setdefault(parent_doc_id, []).append(cacheable)

    ordered_groups = {
        doc_id: sorted(grouped[doc_id], key=_cacheable_source_order)
        for doc_id in sorted(grouped)
    }
    stats = {
        "db_cacheable_occurrences": total_occurrences,
        "duplicate_cacheable_occurrences": duplicate_occurrences,
        "unique_cacheables": len(unique_by_id),
        "empty_cacheables": empty_cacheables,
        "materialized_cacheables": sum(
            len(document_cacheables) for document_cacheables in ordered_groups.values()
        ),
        "documents": len(ordered_groups),
    }
    return ordered_groups, stats


def load_db_cacheables_by_document(
    db_dir: str | Path, batch_size: int = 2048
) -> tuple[dict[str, list[CacheableChunk]], dict[str, int]]:
    """Load the canonical deduplicated cacheable sequence for every DB document."""

    return group_unique_db_cacheables(
        iter_db_cacheables(db_dir=db_dir, batch_size=batch_size)
    )


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
                "DB cacheable requires parent_doc_id for candidate materialization: "
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


def iter_db_cacheable_groups(
    db_dir: str | Path, batch_size: int = 2048
) -> Iterable[tuple[str, list[CacheableChunk]]]:
    """Yield each retrieval chunk ID with its ordered cacheable subchunks."""

    sqlite_path = Path(db_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"missing Chroma sqlite database: {sqlite_path}")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        total_records = connection.execute(
            "select count(*) from embedding_metadata where key='cacheables_json'"
        ).fetchone()[0]
        query = (
            "select e.embedding_id, cacheables.string_value "
            "from embeddings e "
            "join embedding_metadata cacheables "
            "  on e.id = cacheables.id and cacheables.key = 'cacheables_json' "
            "order by e.id "
            "limit ? offset ?"
        )
        for offset in range(0, total_records, batch_size):
            rows = connection.execute(query, (batch_size, offset)).fetchall()
            for chunk_id, cacheables_json in rows:
                payload = json.loads(cacheables_json) if cacheables_json else []
                cacheables = [
                    CacheableChunk.from_payload(item)
                    for item in payload
                    if isinstance(item, dict)
                ]
                yield str(chunk_id), cacheables
    finally:
        connection.close()
