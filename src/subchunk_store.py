"""Row-indexed subchunk text/count storage and Chroma region metadata."""

import json
import os
from functools import lru_cache
from pathlib import Path

from chunk import CacheableChunk

SUBCHUNKS_FILE = "subchunks.json"


@lru_cache(maxsize=1)
def load_subchunks(db_dir: str) -> list[list[str | int]]:
    path = Path(db_dir) / SUBCHUNKS_FILE
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"subchunks must be a row-indexed list: {path}")
    for row, item in enumerate(items):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or type(item[1]) is not int
            or item[1] < 0
        ):
            raise ValueError(f"invalid text/token-count pair at subchunk row {row}")
    return items


def validate_row(row, size: int) -> int:
    if type(row) is not int or not 0 <= row < size:
        raise ValueError(f"invalid subchunk row {row!r} for {size} subchunks")
    return row


def pack_subchunk_range(ids) -> dict:
    if not ids or any(type(row) is not int or row < 0 for row in ids):
        raise ValueError("subchunk range requires nonempty nonnegative integer IDs")
    start = ids[0]
    if any(row != start + index for index, row in enumerate(ids)):
        raise ValueError("subchunk IDs must be consecutive and source ordered")
    return {"subchunk_start": start, "subchunk_end": start + len(ids)}


def pack_regions(specs) -> dict:
    starts, ends = [], []
    for _, indices in specs:
        bounds = pack_subchunk_range(indices)
        starts.append(bounds["subchunk_start"])
        ends.append(bounds["subchunk_end"])
    if not starts:
        raise ValueError("retrieval chunk requires nonempty region membership")
    return {"region_starts": starts, "region_ends": ends}


def cacheables_from_metadata(metadata, subchunks):
    start, end = metadata.get("subchunk_start"), metadata.get("subchunk_end")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(subchunks)
    ):
        raise ValueError("invalid subchunk range")
    parent = metadata.get("parent_doc_id")
    if not isinstance(parent, str) or not parent:
        raise ValueError("retrieval metadata requires parent_doc_id")
    result = []
    for row in range(start, end):
        text, count = subchunks[row]
        result.append(
            CacheableChunk(
                id=row,
                text=text,
                prompt_token_count=count,
                parent_doc_id=parent,
                sentence_ids=[row],
            )
        )
    return result


def unpack_regions(metadata, subchunk_ids):
    start, end = metadata.get("subchunk_start"), metadata.get("subchunk_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("invalid subchunk range")
    if len(subchunk_ids) != end - start or any(
        row != start + index for index, row in enumerate(subchunk_ids)
    ):
        raise ValueError("runtime subchunk order differs from metadata range")
    starts, ends = metadata.get("region_starts"), metadata.get("region_ends")
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or not starts
        or len(starts) != len(ends)
    ):
        raise ValueError("invalid region ranges")
    result = []
    for region_id, (begin, end) in enumerate(zip(starts, ends)):
        if (
            type(begin) is not int
            or type(end) is not int
            or not 0 <= begin < end <= len(subchunk_ids)
        ):
            raise ValueError("invalid region bounds")
        result.append((region_id, tuple(range(begin, end))))
    return result


class SubchunkWriter:
    """Assign IDs once during splitting and stream text/count pairs to disk."""

    def __init__(self, db_dir):
        self.path = Path(db_dir) / SUBCHUNKS_FILE
        self.temporary = self.path.with_suffix(".json.tmp")
        if self.path.exists() or self.temporary.exists():
            raise FileExistsError(f"subchunk store already exists: {self.path}")
        self.handle = self.temporary.open("w", encoding="utf-8")
        self.handle.write("[")
        self.count = 0

    def add_document(self, cacheables):
        for cacheable in cacheables:
            count = cacheable.prompt_token_count
            if type(count) is not int or count < 0:
                raise ValueError("subchunk requires stored prompt_token_count")
            cacheable.id = self.count
            cacheable.sentence_ids = [self.count]
            if self.count:
                self.handle.write(",")
            self.handle.write(
                json.dumps(
                    (cacheable.text, count), ensure_ascii=False, separators=(",", ":")
                )
            )
            self.count += 1

    def finish(self):
        self.handle.write("]")
        self.handle.close()
        os.replace(self.temporary, self.path)
        load_subchunks.cache_clear()

    def close(self):
        self.handle.close()
