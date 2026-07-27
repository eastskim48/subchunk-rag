import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colbert_artifact import DATA_ARTIFACT_FORMAT
from materialize.colbert_materializer import add_region_specs_to_colbert_window_data


class ColBERTSplitMetadataTest(unittest.TestCase):
    def test_finalized_data_reuses_matching_region_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            expected = {
                "format": DATA_ARTIFACT_FORMAT,
                "region_token_budget": 180,
                "region_spec_chunk_count": 3,
            }
            (data_dir / "index.json").write_text(json.dumps(expected), encoding="utf-8")

            actual = add_region_specs_to_colbert_window_data(
                data_dir=data_dir,
                db_dir=data_dir / "db",
                region_token_budget=180,
                overwrite=True,
            )

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
