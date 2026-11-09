"""Custom CORD-19 PMC corpus with existing 512-token sentence splitter; no embeddings."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-02-15/document_parses.tar.gz"
SIZE = 4289907198


def dump(path, data):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def line(file, data):
    file.write(json.dumps(data, ensure_ascii=False) + "\n")


def download(root, status):
    target = root / "source/document_parses.tar.gz"
    if target.exists():
        assert target.stat().st_size == SIZE
        return target
    parts = root / "source/download_parts"
    parts.mkdir(exist_ok=True)
    width = (SIZE + 15) // 16

    def fetch(i):
        start, end = i * width, min((i + 1) * width, SIZE) - 1
        path = parts / str(i)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    URL, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"].startswith(
                        f"bytes {start}-{end}/"
                    )
                    with path.open("wb") as out:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                assert path.stat().st_size == end - start + 1
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        jobs = [pool.submit(fetch, i) for i in range(16)]
        while not all(job.done() for job in jobs):
            status(
                stage="download",
                downloaded_bytes=sum(p.stat().st_size for p in parts.iterdir()),
                total_bytes=SIZE,
            )
            time.sleep(10)
        for job in jobs:
            job.result()
    temporary = target.with_suffix(".assembling")
    sha = hashlib.sha256()
    with temporary.open("wb") as out:
        for i in range(16):
            with (parts / str(i)).open("rb") as inp:
                while block := inp.read(4 * 1024 * 1024):
                    out.write(block)
                    sha.update(block)
    assert temporary.stat().st_size == SIZE
    temporary.replace(target)
    dump(
        root / "source/archive_manifest.json",
        {"url": URL, "bytes": SIZE, "sha256": sha.hexdigest()},
    )
    for i in range(16):
        (parts / str(i)).unlink()
    parts.rmdir()
    return target


def init_worker(documents):
    global splitter
    from transformers import AutoTokenizer
    from materialize.splitter.base import SentenceWiseSplitter

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", revision=REVISION, local_files_only=True
    )
    splitter = SentenceWiseSplitter(
        str(documents), tokenizer, None, 512, max_subchunk_tokens=180
    )


def split(name):
    result = splitter.split_document(name)
    return (
        name,
        [vars(c) for c in result.chunks],
        [
            {
                "id": c.id,
                "text": c.text,
                "token_count": c.token_count,
                "metadata": c.metadata,
                "sentence_ids": [s.id for s in c.cacheables],
            }
            for c in result.retrievable_chunks
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output_dir
    for name in (
        "source",
        "documents",
        "questions",
        "answers",
        "dataset_info",
        "preprocessed512",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)

    def status(**values):
        dump(
            root / "status.json",
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **values,
            },
        )
        print(json.dumps(values), flush=True)

    try:
        rows = [
            json.loads(x)
            for x in (root / "source/dense_test.txt").read_text().splitlines()
        ]
        with (
            (root / "questions/query.jsonl").open("w") as q,
            (root / "answers/answer.jsonl").open("w") as a,
            (root / "dataset_info/retrieval_test.jsonl").open("w") as e,
        ):
            for i, row in enumerate(rows):
                line(q, {"id": i, "query": row["question"]})
                line(a, {"id": i, "answers": row["answers"]})
                line(
                    e,
                    {
                        "id": i,
                        "source_row": i,
                        "document_id": row["id"],
                        "source_record": row,
                    },
                )
        archive = download(root, status)
        names = []
        seen = set()
        with (
            tarfile.open(archive, "r|gz") as tar,
            (root / "dataset_info/documents.jsonl").open("w") as mapping,
        ):
            for member in tar:
                if (
                    not member.isfile()
                    or "/pmc_json/" not in member.name
                    or not member.name.endswith(".json")
                ):
                    continue
                with tar.extractfile(member) as inp:
                    doc = json.load(inp)
                name = Path(member.name).stem + ".txt"
                if name in seen:
                    raise ValueError(f"Duplicate document filename: {name}")
                seen.add(name)
                text = "\n\n".join(
                    [doc["metadata"]["title"]] + [p["text"] for p in doc["body_text"]]
                )
                (root / "documents" / name).write_text(text, encoding="utf-8")
                names.append(name)
                line(
                    mapping,
                    {
                        "filename": name,
                        "paper_id": doc["paper_id"],
                        "archive_member": member.name,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    },
                )
                if len(names) % 1000 == 0:
                    status(stage="extract_pmc_documents", documents=len(names))
        if not names:
            raise ValueError("No pmc_json documents found; corpus construction stopped")
        counts = {"documents": 0, "sentences": 0, "chunks": 0}
        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(root / "documents",),
            ) as pool,
            (root / "preprocessed512/sentences.jsonl").open("w") as sentences,
            (root / "preprocessed512/chunks.jsonl").open("w") as chunks,
        ):
            for name, units, windows in pool.map(split, names, chunksize=8):
                for unit in units:
                    line(sentences, unit)
                for window in windows:
                    line(chunks, window)
                counts["documents"] += 1
                counts["sentences"] += len(units)
                counts["chunks"] += len(windows)
                if counts["documents"] % 100 == 0:
                    status(stage="split512", total_documents=len(names), **counts)
        dump(
            root / "manifest.json",
            {
                "custom_configuration": True,
                "corpus": "CORD-19 2021-02-15 pmc_json, as official script input",
                "text": "title + body_text paragraphs joined by two newlines; no paragraph filtering",
                "question_source": "Open_Domain_COVIDQA c7ac04d7cdcdf36cd5491455a5b3353e9f3e9b9f data/dense_test.txt; original row order and all answers",
                "questions": len(rows),
                "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
                "tokenizer_revision": REVISION,
                "retrievable_chunk_size": 512,
                "stride": 512,
                "max_subchunk_tokens": 180,
                "splitter": "materialize.splitter.base.SentenceWiseSplitter",
                "embeddings_created": False,
                **counts,
            },
        )
        status(stage="complete", questions=len(rows), **counts)
    except Exception as exc:
        status(stage="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
