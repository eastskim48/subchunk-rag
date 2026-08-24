import itertools
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from materialize.colbert_materializer import (  # noqa: E402
    ColBERTWindowEncoder,
    WindowSpec,
    _build_fixed_chunk_window_spec,
)


class ColBERTContiguousWindowTest(unittest.TestCase):
    def test_fixed_chunk_spec_uses_the_db_center_without_context(self):
        spec = _build_fixed_chunk_window_spec(
            center_unit="fixed_chunk",
            source_tokenizer=None,
            visible_token_overhead=2,
            source_token_ids=[],
            chunk_start=4,
            chunk_end=6,
            center_text="CENTER",
            center_index=2,
            window_token_budget=10,
        )

        self.assertEqual(spec.text, "CENTER")
        self.assertEqual((spec.center_start, spec.center_end), (0, 6))
        self.assertEqual(spec.selected_indices, [2])

    def test_fixed_chunk_window_spec_preserves_center_span(self):
        class FakeTokenizer:
            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(token_id) for token_id in token_ids)

        spec = _build_fixed_chunk_window_spec(
            center_unit="fixed_chunk_window",
            source_tokenizer=FakeTokenizer(),
            visible_token_overhead=2,
            source_token_ids=list(range(10)),
            chunk_start=4,
            chunk_end=6,
            center_text="CENTER",
            center_index=2,
            window_token_budget=10,
        )

        self.assertEqual(spec.text, "1 2 3 CENTER 6 7 8")
        self.assertEqual(
            spec.text[spec.center_start : spec.center_end],
            "CENTER",
        )
        self.assertEqual(spec.selected_indices, [2])

    def test_window_encoder_delegates_text_and_center_spans(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        captured = {}

        def encode_document_spans(texts, center_spans, show_progress=False):
            captured["texts"] = texts
            captured["center_spans"] = center_spans
            captured["show_progress"] = show_progress
            return ["encoded"]

        encoder.encode_document_spans = encode_document_spans
        specs = [
            WindowSpec(
                text="left center right",
                center_start=5,
                center_end=11,
                selected_indices=[0, 1, 2],
                addition_order=[1, 0, 2],
                truncated_center=False,
            )
        ]

        result = encoder.encode_windows(specs, show_progress=True)

        self.assertEqual(result, ["encoded"])
        self.assertEqual(captured["texts"], ["left center right"])
        self.assertEqual(captured["center_spans"], [(5, 11)])
        self.assertTrue(captured["show_progress"])

    def test_encoder_window_budget_overflow_stops_expansion(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        encoder.doc_maxlen = 40
        encoder.doc_token_overhead = 0
        encoder.token_counts_without_specials = lambda sentences: [
            int(sentence) for sentence in sentences
        ]

        specs = encoder.build_centered_windows(
            ["10", "1000", "10", "10", "10"],
            window_token_budget=40,
        )

        self.assertEqual(specs[2].selected_indices, [2])
        self.assertEqual(specs[2].addition_order, [2])

    def test_region_budget_can_exceed_encoder_document_limit(self):
        encoder = object.__new__(ColBERTWindowEncoder)
        encoder.doc_maxlen = 40
        encoder.doc_token_overhead = 0
        encoder.token_counts_without_specials = lambda sentences: [
            int(sentence) for sentence in sentences
        ]

        specs = encoder.build_centered_windows(
            ["30", "30", "30"],
            window_token_budget=90,
        )

        self.assertEqual(specs[1].selected_indices, [0, 1, 2])

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
                            window_token_budget=token_budget,
                        )
                        for spec in specs:
                            selected = spec.selected_indices
                            self.assertEqual(
                                selected,
                                list(range(selected[0], selected[-1] + 1)),
                            )


if __name__ == "__main__":
    unittest.main()
