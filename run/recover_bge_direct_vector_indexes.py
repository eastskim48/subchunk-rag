"""Build and independently verify vector-only replacements, then retire old DBs."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from build_bge_direct_regions import read, save


def cleanup(spec):
    artifact = Path(spec["artifact_dir"]).resolve()
    old = Path(spec["old_db"])
    new = Path(spec["output"]).resolve()
    manifest = read(new / "index_complete.json")
    assert manifest["storage_layout"] == "vectors_and_ids_only"
    assert manifest["source_artifact"] == str(artifact)
    assert manifest["verification"]["fulltext_rows"] == 0
    assert old == artifact / "db" and not old.is_symlink()
    assert old.resolve() != Path(spec["source_db"]).resolve()
    assert old.resolve() != new / "db"
    assert (artifact / "encoded.json").is_file()
    marker = new / "old_db_removed.json"
    if marker.exists():
        return
    existed = old.exists()
    if existed:
        shutil.rmtree(old)
    assert not old.exists()
    assert (artifact / "embeddings.float32.bin").is_file()
    assert (artifact / "regions.sqlite3").is_file()
    save(marker, dict(old_db=str(old), existed=existed, removed_at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    logs = Path(cfg["logs"])
    failures = []
    started = time.time()
    for name, spec in cfg["datasets"].items():
        if (Path(spec["output"]) / "index_complete.json").exists():
            continue
        for phase in ["index", "verify"]:
            save(
                logs / "status.json", dict(stage=phase, dataset=name, failures=failures)
            )
            with (logs / f"{name}.{phase}.log").open("a") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(
                            Path(__file__).with_name("build_bge_direct_vector_index.py")
                        ),
                        "--config",
                        str(args.config),
                        "--dataset",
                        name,
                        "--phase",
                        phase,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                failures.append(
                    dict(dataset=name, phase=phase, returncode=result.returncode)
                )
                break
    if not failures:
        for name, spec in cfg["datasets"].items():
            save(logs / "status.json", dict(stage="cleanup", dataset=name, failures=[]))
            try:
                cleanup(spec)
            except Exception as error:
                failures.append(dict(dataset=name, phase="cleanup", error=str(error)))
    save(
        logs / "status.json",
        dict(
            stage="complete_with_failures" if failures else "complete",
            failures=failures,
            started=started,
            finished=time.time(),
        ),
    )


if __name__ == "__main__":
    main()
