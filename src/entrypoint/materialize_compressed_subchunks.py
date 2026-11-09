"""Encode ColBERT subchunks to FP16, then convert to 2-bit residual storage."""

import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize.compressed_colbert_materializer import (  # noqa: E402
    build_compressed_colbert_subchunks,
)

if __name__ == "__main__":
    fire.Fire(build_compressed_colbert_subchunks)
