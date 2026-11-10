"""File handling for the existing custom dataset preparation entrypoints."""

import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_file(
    url: str, destination: Path, expected_sha256=None, *, reject_html=False
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_name(destination.name + ".part")
        request = Request(url, headers={"User-Agent": "subchunk-dataset-preparer/1"})
        with urlopen(request, timeout=120) as response:
            if reject_html and (
                response.headers.get_content_type() == "text/html"
                or "/accounts/login" in urlsplit(response.geturl()).path
            ):
                raise PermissionError(
                    "The server returned a login/HTML page instead of an archive. "
                    "Check the source URL or supply a downloaded local archive."
                )
            with partial.open("xb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
        if expected_sha256 and file_sha256(partial) != expected_sha256:
            raise ValueError(f"source checksum mismatch; retained {partial}")
        partial.rename(destination)
    if expected_sha256 and file_sha256(destination) != expected_sha256:
        raise ValueError(f"source checksum mismatch: {destination}")
    return destination


def staging_directory(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
