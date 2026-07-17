import itertools
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from materialize.colbert_window import (  # noqa: E402
    ColBERTWindowEncoder,
    _centered_region_index_specs,
)


class ColBERTContiguousWindowTest(unittest.TestCase):
    def test_encoder_window_budget_overflow_stops_expansion(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        encoder.doc_maxlen = 40
        encoder.doc_token_overhead = 0
        encoder.token_counts_without_specials = lambda sentences: [
            int(sentence) for sentence in sentences
        ]

        specs = encoder.build_centered_windows(
            ["10", "1000", "10", "10", "10"],
            token_budget=40,
        )

        self.assertEqual(specs[2].selected_indices, [2])
        self.assertEqual(specs[2].addition_order, [2])

    def test_region_spec_budget_overflow_stops_instead_of_filling_opposite_side(self):
        specs = _centered_region_index_specs(
            [10, 1000, 10, 10, 10],
            token_budget=40,
            doc_token_overhead=0,
        )

        self.assertIn((2, (2,)), specs)
        self.assertNotIn((2, (2, 3, 4)), specs)

    def test_generated_region_specs_are_contiguous(self):
        for length in range(1, 8):
            for token_counts in itertools.product([1, 2, 5, 20, 100], repeat=length):
                for token_budget in [3, 5, 8, 12, 25, 40]:
                    for doc_token_overhead in [0, 2]:
                        with self.subTest(
                            token_counts=token_counts,
                            token_budget=token_budget,
                            doc_token_overhead=doc_token_overhead,
                        ):
                            specs = _centered_region_index_specs(
                                list(token_counts),
                                token_budget=token_budget,
                                doc_token_overhead=doc_token_overhead,
                            )
                            for _, selected in specs:
                                self.assertEqual(
                                    selected,
                                    tuple(range(selected[0], selected[-1] + 1)),
                                )

    def test_generated_encoder_windows_are_contiguous(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        encoder.doc_maxlen = 1000
        encoder.doc_token_overhead = 0

        for length in range(1, 8):
            sentences = [str(idx) for idx in range(length)]
            for token_counts in itertools.product([1, 2, 5, 20, 100], repeat=length):
                for token_budget in [3, 5, 8, 12, 25, 40, 120]:
                    with self.subTest(
                        token_counts=token_counts,
                        token_budget=token_budget,
                    ):
                        encoder.token_counts_without_specials = (
                            lambda values, counts=token_counts: list(counts)
                        )
                        specs = encoder.build_centered_windows(
                            sentences,
                            token_budget=token_budget,
                        )
                        for spec in specs:
                            selected = spec.selected_indices
                            self.assertEqual(
                                selected,
                                list(range(selected[0], selected[-1] + 1)),
                            )


if __name__ == "__main__":
    unittest.main()
