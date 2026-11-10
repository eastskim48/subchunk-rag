"""Prepare the existing BioASQ14b non-summary/linked-abstract configuration."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

from dataset.bioasq.questions import prepare as prepare_questions
from dataset.preparation import (
    file_sha256,
    staging_directory,
    write_json,
    write_jsonl,
)

SOURCE_MEMBER = "BioASQ-training14b/training14b.json"
SOURCE_SHA256 = "5669afbc0bbc8f50d54850edb4aa592ecff3e38b29b09ac6684bb52990b651b7"
EXPECTED_PUBMED_IDS = 58_048
EXPECTED_DOCUMENTS = 58_012


def fetch_pubmed(ids: list[str], cache: Path) -> Path:
    """Reuse the original batching, XML parsing, and returned-record order."""
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "pubmed_records.jsonl"
    if destination.exists():
        return destination
    request_lock = threading.Lock()
    last_request = [0.0]

    def fetch_batch(item):
        index, start = item
        path = cache / "pubmed_xml" / f"{index:04d}.xml.gz"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            parameters = urlencode(
                {
                    "db": "pubmed",
                    "id": ",".join(ids[start : start + 200]),
                    "retmode": "xml",
                }
            )
            with request_lock:
                time.sleep(max(0.0, 0.4 - (time.monotonic() - last_request[0])))
                last_request[0] = time.monotonic()
            with urlopen(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                + parameters,
                timeout=60,
            ) as response:
                payload = response.read()
            if ET.fromstring(payload).tag != "PubmedArticleSet":
                raise ValueError(f"unexpected PubMed response for batch {index}")
            path.write_bytes(gzip.compress(payload))
        return index, path

    records = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, path in executor.map(
            fetch_batch, enumerate(range(0, len(ids), 200))
        ):
            root = ET.fromstring(gzip.decompress(path.read_bytes()))
            for node in root:
                if node.tag == "PubmedArticle":
                    pmid = node.findtext("MedlineCitation/PMID")
                    article = node.find("MedlineCitation/Article")
                elif node.tag == "PubmedBookArticle":
                    pmid = node.findtext("BookDocument/PMID")
                    article = node.find("BookDocument")
                else:
                    raise ValueError(f"Unexpected PubMed XML record: {node.tag}")
                title_node = article.find("ArticleTitle")
                title = "" if title_node is None else "".join(title_node.itertext())
                abstract = [
                    "".join(n.itertext())
                    for n in article.findall("Abstract/AbstractText")
                ]
                records[pmid] = dict(pmid=pmid, title=title, abstract_sections=abstract)
            if index % 20 == 0:
                print(f"PubMed batch {index}: {len(records)} records", flush=True)
    write_jsonl(destination, records.values())
    write_json(
        cache / "pubmed_download.json",
        {
            "requested_ids": len(ids),
            "fetched_records": len(records),
            "missing_ids": sorted(set(ids) - set(records)),
            "unexpected_ids": sorted(set(records) - set(ids)),
        },
    )
    return destination


def prepare(source: Path, records: Path, output: Path) -> dict:
    staging = staging_directory(output)
    work = staging / "dataset"
    prepare_questions(source, work)
    docs = work / "documents"
    docs.mkdir()
    seen = set()
    with records.open() as handle:
        for line in handle:
            row = json.loads(line)
            pmid = row["pmid"]
            if not pmid.isdecimal() or pmid in seen:
                raise ValueError(f"invalid or duplicate PubMed ID: {pmid!r}")
            seen.add(pmid)
            text = "\n".join([row["title"], *row["abstract_sections"]])
            (docs / f"{pmid}.txt").write_text(text, encoding="utf-8")
    if len(seen) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"existing custom corpus requires {EXPECTED_DOCUMENTS} records, got {len(seen)}; "
            "source records retained for inspection"
        )
    manifest = {
        "custom_protocol": True,
        "scope": "Existing BioASQ14b training non-summary questions and all available linked title/abstract documents",
        "documents": len(seen),
        "questions": json.loads((work / "manifest.json").read_text())[
            "selected_questions"
        ],
        "source_question_sha256": file_sha256(source),
        "source_pubmed_records": str(records),
        "source_pubmed_records_sha256": file_sha256(records),
        "corpus_references": "All original 5729 questions, including summary questions",
        "question_selection": "Existing summary-only exclusion; all selected annotations preserved",
        "document_text": "title and abstract_sections joined by newline, unchanged",
    }
    write_json(work / "dataset_info/manifest.json", manifest)
    if output.exists():
        raise FileExistsError(output)
    work.rename(output)
    staging.rmdir()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Original training14b ZIP downloaded manually from the BioASQ participant area",
    )
    parser.add_argument("--pubmed-records", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = (args.source_dir or output.with_name(output.name + "-sources")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive.resolve()) as archive:
        payload = archive.read(SOURCE_MEMBER)
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("archive does not contain the existing training14b source")
    source = cache / "training14b.json"
    if source.exists() and source.read_bytes() != payload:
        raise ValueError(f"different source already exists: {source}")
    source.write_bytes(payload)
    questions = json.loads(payload)["questions"]
    ids = sorted(
        {url.rstrip("/").split("/")[-1] for q in questions for url in q["documents"]}
    )
    if len(ids) != EXPECTED_PUBMED_IDS or not all(pmid.isdecimal() for pmid in ids):
        raise ValueError(
            "source linked-document IDs differ from the existing configuration"
        )
    if args.pubmed_records is None:
        records = fetch_pubmed(ids, cache)
    else:
        records = args.pubmed_records.resolve()
    manifest = prepare(source, records, output)
    print(
        f"Prepared {manifest['questions']} questions and {manifest['documents']} documents"
    )


if __name__ == "__main__":
    main()
