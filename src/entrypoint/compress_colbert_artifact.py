"""CLI entrypoint for rollback-safe ColBERT 2-bit artifact conversion."""

import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize.colbert_residual import compress_colbert_window_artifact  # noqa: E402


def main(
    source_dir: str,
    output_dir: str,
    num_centroids: int | None = None,
    sample_size: int | None = None,
    seed: int = 123,
    kmeans_niters: int = 4,
    batch_tokens: int = 4096,
    use_gpu: bool = True,
    storage_layout: str = "interleaved",
):
    summary = compress_colbert_window_artifact(
        source_dir=source_dir,
        output_dir=output_dir,
        num_centroids=num_centroids,
        sample_size=sample_size,
        seed=seed,
        kmeans_niters=kmeans_niters,
        batch_tokens=batch_tokens,
        use_gpu=use_gpu,
        storage_layout=storage_layout,
    )
    print(f"ColBERT 2-bit residual artifact materialized: {summary}")


if __name__ == "__main__":
    fire.Fire(main)
