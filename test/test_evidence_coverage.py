import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_coverage import (
    TextEvidenceCoverageScorer,
    _longest_common_substring_length,
    load_text_evidence_labels,
    summarize_text_evidence_records,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


class TextEvidenceCoverageTest(unittest.TestCase):
    def test_loads_labels_without_source_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.jsonl"
            path.write_text(
                "\n".join(
                    [
                        (
                            '{"query":"q1","evidence_passage_ids":["0-0"],'
                            '"evidence_texts":["a"]}'
                        ),
                        (
                            '{"query":"q2","evidence_passage_ids":["1-0"],'
                            '"evidence_texts":["b"]}'
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            labels = load_text_evidence_labels(path)

            self.assertEqual(set(labels), {"q1", "q2"})

    def test_longest_common_substring_requires_contiguity(self):
        self.assertEqual(
            _longest_common_substring_length("abcdefgh", "a1b2c3d4e5f6g7h"),
            1,
        )
        self.assertEqual(
            _longest_common_substring_length("abcdefgh", "xxcdefyy"),
            4,
        )
        self.assertEqual(
            _longest_common_substring_length([1, 2, 3], [0, 1, 2, 3, 4]),
            3,
        )

    def test_exact_containment_is_primary_and_partial_match_is_secondary(self):
        label = {
            "query": "query",
            "evidence_passage_ids": ["0-0"],
            "evidence_texts": ["cdefgh"],
        }
        scorer = TextEvidenceCoverageScorer(
            metric_tokenizer=CharacterTokenizer(),
            passage_recall_threshold=0.8,
        )

        score = scorer.score(
            label=label,
            retrieved_context="prefix cdefgh suffix",
            compressed_context="xxcdefyy",
        )

        self.assertEqual(score["retrieval"]["evidence_char_exact_recall"], 1.0)
        self.assertEqual(score["retrieval"]["evidence_token_exact_recall"], 1.0)
        self.assertEqual(score["compressed"]["evidence_char_exact_recall"], 0.0)
        self.assertEqual(score["compressed"]["evidence_token_exact_recall"], 0.0)
        self.assertAlmostEqual(
            score["compressed"]["evidence_char_partial_recall"], 4 / 6
        )
        self.assertAlmostEqual(
            score["compressed"]["evidence_token_partial_recall"], 4 / 7
        )
        self.assertEqual(score["conditional"]["evidence_char_exact_retention"], 0.0)
        self.assertAlmostEqual(
            score["conditional"]["evidence_char_partial_retention"], 4 / 6
        )

        score["retrieval"]["context_tokens"] = 20
        score["compressed"]["context_tokens"] = 8
        summary = summarize_text_evidence_records([score], passage_recall_threshold=0.8)
        self.assertEqual(summary["primary_metric"], "evidence_char_exact_recall")
        self.assertEqual(summary["retrieval"]["all_evidence_char_exact"], 1.0)
        self.assertEqual(summary["compressed"]["all_evidence_char_exact"], 0.0)

    def test_scattered_characters_do_not_count_as_exact_or_partial_sequence(self):
        label = {
            "query": "query",
            "evidence_passage_ids": ["0-0"],
            "evidence_texts": ["abcdefgh"],
        }
        scorer = TextEvidenceCoverageScorer(CharacterTokenizer())

        score = scorer.score(
            label=label,
            retrieved_context="a1b2c3d4e5f6g7h",
            compressed_context="a1b2c3d4e5f6g7h",
        )

        self.assertEqual(score["retrieval"]["evidence_char_exact_recall"], 0.0)
        self.assertEqual(score["retrieval"]["evidence_token_exact_recall"], 0.0)
        self.assertEqual(score["retrieval"]["evidence_char_partial_recall"], 1 / 8)
        self.assertEqual(score["retrieval"]["evidence_token_partial_recall"], 2 / 9)

    def test_whitespace_runs_are_normalized_but_inserted_separator_is_not_removed(self):
        label = {
            "query": "query",
            "evidence_passage_ids": ["0-0"],
            "evidence_texts": ["small  pox"],
        }
        scorer = TextEvidenceCoverageScorer(CharacterTokenizer())

        normalized = scorer.score(
            label=label,
            retrieved_context="small\n\tpox",
            compressed_context="small\n\tpox",
        )
        separated = scorer.score(
            label={
                "query": "query",
                "evidence_passage_ids": ["0-0"],
                "evidence_texts": ["smallpox"],
            },
            retrieved_context="small pox",
            compressed_context="small pox",
        )

        self.assertEqual(normalized["retrieval"]["evidence_char_exact_recall"], 1.0)
        self.assertEqual(separated["retrieval"]["evidence_char_exact_recall"], 0.0)

    def test_multiple_passages_report_passage_and_all_evidence_exact_recall(self):
        label = {
            "query": "query",
            "evidence_passage_ids": ["0-0", "0-1"],
            "evidence_texts": ["alpha", "beta"],
        }
        scorer = TextEvidenceCoverageScorer(CharacterTokenizer())

        score = scorer.score(
            label=label,
            retrieved_context="alpha only",
            compressed_context="alpha only",
        )

        self.assertEqual(score["retrieval"]["evidence_char_exact_recall"], 0.5)
        self.assertEqual(score["retrieval"]["any_evidence_char_exact"], 1.0)
        self.assertEqual(score["retrieval"]["all_evidence_char_exact"], 0.0)

    def test_source_metadata_is_not_required_or_checked(self):
        label = {
            "query": "query",
            "document_file": "missing.txt",
            "evidence_passage_ids": ["0-0"],
            "evidence_texts": ["evidence"],
            "evidence_char_spans": [[999, 1007]],
        }
        scorer = TextEvidenceCoverageScorer(CharacterTokenizer())

        score = scorer.score(
            label=label,
            retrieved_context="the evidence is visible",
            compressed_context="evidence",
        )

        self.assertEqual(score["retrieval"]["evidence_char_exact_recall"], 1.0)
        self.assertEqual(score["compressed"]["evidence_char_exact_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
