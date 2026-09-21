"""Disk-backed metadata for ColBERT candidate-store artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1
SQLITE_LOOKUP_BATCH_SIZE = 900


def _encode_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ColBERTMetadataWriter:
    """Incrementally write cacheable and retrieval-region metadata."""

    def __init__(self, path: str | Path, *, create: bool = True):
        self.path = Path(path)
        if create and self.path.exists():
            raise FileExistsError(f"ColBERT metadata database exists: {self.path}")
        if not create and not self.path.exists():
            raise FileNotFoundError(f"missing ColBERT metadata database: {self.path}")
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("pragma journal_mode=wal")
        self.connection.execute("pragma synchronous=normal")
        if create:
            self.connection.execute(f"pragma user_version={SCHEMA_VERSION}")
            self.connection.executescript("""
                create table cacheables (
                    cacheable_id text primary key,
                    row_index integer not null unique,
                    window_ids_json text not null
                );
                create table regions (
                    chunk_id text primary key,
                    cacheable_ids_json text not null,
                    specs_json text not null
                );
                """)
            self.connection.commit()
        else:
            schema_version = int(
                self.connection.execute("pragma user_version").fetchone()[0]
            )
            if schema_version != SCHEMA_VERSION:
                raise ValueError(
                    "unsupported ColBERT metadata schema version: "
                    f"{schema_version}; expected {SCHEMA_VERSION}"
                )

    def add_cacheables(
        self,
        records: Iterable[tuple[str, int, Sequence[str]]],
    ) -> None:
        payloads = [
            (str(cacheable_id), int(row_index), _encode_json(list(window_ids)))
            for cacheable_id, row_index, window_ids in records
        ]
        try:
            self.connection.executemany(
                "insert into cacheables "
                "(cacheable_id, row_index, window_ids_json) values (?, ?, ?)",
                payloads,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate cacheable ID or row in ColBERT metadata"
            ) from exc
        self.connection.commit()

    def replace_regions(
        self,
        records: Iterable[
            tuple[str, Sequence[str], Sequence[tuple[int, Sequence[int]]]]
        ],
    ) -> None:
        payloads = [
            (
                str(chunk_id),
                _encode_json(list(cacheable_ids)),
                _encode_json(
                    [
                        [int(center_idx), [int(index) for index in selected_indices]]
                        for center_idx, selected_indices in specs
                    ]
                ),
            )
            for chunk_id, cacheable_ids, specs in records
        ]
        self.connection.executemany(
            "insert or replace into regions "
            "(chunk_id, cacheable_ids_json, specs_json) values (?, ?, ?)",
            payloads,
        )
        self.connection.commit()

    def clear_regions(self) -> None:
        self.connection.execute("delete from regions")
        self.connection.commit()

    def close(self) -> None:
        if self.connection is None:
            return
        self.connection.execute("pragma wal_checkpoint(truncate)")
        self.connection.execute("pragma journal_mode=delete")
        self.connection.close()
        self.connection = None


class ColBERTMetadataReader:
    """Read only the metadata needed for the current retrieval result."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"missing ColBERT metadata database: {self.path}")
        self.connection = sqlite3.connect(
            f"file:{self.path}?mode=ro&immutable=1",
            uri=True,
        )
        self.connection.execute("pragma query_only=on")
        self.connection.execute("pragma cache_size=-32768")
        schema_version = int(
            self.connection.execute("pragma user_version").fetchone()[0]
        )
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                "unsupported ColBERT metadata schema version: "
                f"{schema_version}; expected {SCHEMA_VERSION}"
            )
        table_names = {
            str(row[0])
            for row in self.connection.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        if not {"cacheables", "regions"}.issubset(table_names):
            raise ValueError("ColBERT metadata database is missing required tables")

    def rows_for_cacheable_ids(
        self, cacheable_ids: Sequence[str | None]
    ) -> list[int | None]:
        normalized_ids = [
            str(cacheable_id)
            for cacheable_id in cacheable_ids
            if cacheable_id is not None
        ]
        if not normalized_ids:
            return [None] * len(cacheable_ids)
        rows_by_id = {}
        for start in range(0, len(normalized_ids), SQLITE_LOOKUP_BATCH_SIZE):
            lookup_ids = normalized_ids[start : start + SQLITE_LOOKUP_BATCH_SIZE]
            placeholders = ",".join("?" for _ in lookup_ids)
            rows_by_id.update(
                {
                    str(cacheable_id): int(row_index)
                    for cacheable_id, row_index in self.connection.execute(
                        "select cacheable_id, row_index from cacheables "
                        f"where cacheable_id in ({placeholders})",
                        lookup_ids,
                    )
                }
            )
        return [
            rows_by_id.get(str(cacheable_id)) if cacheable_id is not None else None
            for cacheable_id in cacheable_ids
        ]

    def window_ids_for_cacheable_ids(
        self, cacheable_ids: Sequence[str | None]
    ) -> list[list[str]]:
        normalized_ids = [
            str(cacheable_id)
            for cacheable_id in cacheable_ids
            if cacheable_id is not None
        ]
        if not normalized_ids:
            return [[] for _ in cacheable_ids]
        windows_by_id = {}
        for start in range(0, len(normalized_ids), SQLITE_LOOKUP_BATCH_SIZE):
            lookup_ids = normalized_ids[start : start + SQLITE_LOOKUP_BATCH_SIZE]
            placeholders = ",".join("?" for _ in lookup_ids)
            windows_by_id.update(
                {
                    str(cacheable_id): [
                        str(item) for item in json.loads(window_ids_json)
                    ]
                    for cacheable_id, window_ids_json in self.connection.execute(
                        "select cacheable_id, window_ids_json from cacheables "
                        f"where cacheable_id in ({placeholders})",
                        lookup_ids,
                    )
                }
            )
        return [
            (
                windows_by_id.get(str(cacheable_id), [str(cacheable_id)])
                if cacheable_id is not None
                else []
            )
            for cacheable_id in cacheable_ids
        ]

    def iter_cacheable_ids(self, batch_size: int = 8192):
        for cacheable_id, _ in self.iter_cacheable_rows(batch_size=batch_size):
            yield cacheable_id

    def iter_cacheable_rows(self, batch_size: int = 8192):
        cursor = self.connection.execute(
            "select cacheable_id, row_index from cacheables order by row_index"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for cacheable_id, row_index in rows:
                yield str(cacheable_id), int(row_index)

    def iter_region_payloads(self, batch_size: int = 8192):
        cursor = self.connection.execute(
            "select chunk_id, cacheable_ids_json, specs_json from regions"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for chunk_id, cacheable_ids_json, specs_json in rows:
                yield str(chunk_id), {
                    "cacheable_ids": [
                        str(item) for item in json.loads(cacheable_ids_json)
                    ],
                    "specs": json.loads(specs_json),
                }

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
