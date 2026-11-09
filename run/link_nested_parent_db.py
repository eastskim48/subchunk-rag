"""Create a separate child DB copy with verified, unique parent chunk links."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-child-db", type=Path, required=True)
    parser.add_argument("--child-db", type=Path, required=True)
    parser.add_argument("--parent-db", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    if args.child_db.exists():
        raise FileExistsError(args.child_db)
    import chromadb

    shutil.copytree(args.source_child_db, args.child_db)
    child = chromadb.PersistentClient(path=str(args.child_db)).get_collection(
        "doc_collection", embedding_function=None
    )
    parent = chromadb.PersistentClient(path=str(args.parent_db)).get_collection(
        "doc_collection", embedding_function=None
    )
    parents = parent.get(include=["metadatas", "documents"])
    by_start = {
        (m["parent_doc_id"], m["source_token_start"]): (cid, m, text)
        for cid, m, text in zip(
            parents["ids"], parents["metadatas"], parents["documents"]
        )
    }
    children = child.get(include=["metadatas", "documents"])
    links, groups = {}, {}
    for cid, m, text in zip(
        children["ids"], children["metadatas"], children["documents"]
    ):
        parent_start = m["source_token_start"] // 512 * 512
        pid, pm, ptext = by_start[m["parent_doc_id"], parent_start]
        assert (
            pm["source_token_start"]
            <= m["source_token_start"]
            < m["source_token_end"]
            <= pm["source_token_end"]
        )
        assert (
            pm["source_char_start"]
            <= m["source_char_start"]
            < m["source_char_end"]
            <= pm["source_char_end"]
        )
        begin = m["source_char_start"] - pm["source_char_start"]
        assert ptext[begin : begin + len(text)] == text
        links[cid] = pid
        groups.setdefault(pid, []).append((m["source_char_start"], text))
    assert len(groups) == parent.count()
    for pid, parts in groups.items():
        ptext = next(
            text
            for cid, text in zip(parents["ids"], parents["documents"])
            if cid == pid
        )
        assert "".join(text for _, text in sorted(parts)) == ptext
    for start in range(0, len(children["ids"]), 1000):
        ids = children["ids"][start : start + 1000]
        child.update(
            ids=ids, metadatas=[{"parent_chunk_id": links[cid]} for cid in ids]
        )
    manifest_path = args.child_db / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        nested_parent_size=512,
        parent_db_dir=str(args.parent_db),
        construction="exact128 subdivisions of512 parent; copied identical aligned BGE128 texts/vectors/graph",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "child_db_dir": str(args.child_db.resolve()),
        "parent_db_dir": str(args.parent_db.resolve()),
        "max_parents": 5,
        "links": links,
        "validation": {
            "children": len(links),
            "parents": len(groups),
            "unique_parent_per_child": True,
            "children_reconstruct_every_parent_exactly": True,
        },
    }
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["validation"]), flush=True)


if __name__ == "__main__":
    main()
