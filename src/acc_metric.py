"""Dataset-aware exact-match and token-F1 evaluation utilities."""

import json
import itertools
import math
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Optional

ANSWER_PREFIX_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)
QUESTION_SPLIT_RE = re.compile(r"\bquestion\s*:\s*", re.IGNORECASE)
TRAILING_SPLIT_PATTERNS = (
    re.compile(r"\(\s*note\s*[:)]", re.IGNORECASE),
    re.compile(r"\bnote\s*:", re.IGNORECASE),
    re.compile(r"\bsource\s*:", re.IGNORECASE),
    re.compile(r"\bhowever\b", re.IGNORECASE),
)
DOUBLE_SPACE_SENTENCE_CONTINUATION_RE = re.compile(r"([.!?])\s{2,}[A-Z0-9\"']")
REPEATED_SENTENCE_RE = re.compile(
    r"^\s*(?P<fragment>[^.?!]+[.?!])(?:\s+(?P=fragment))+\s*$"
)
TRIVIA_EXTRA_PUNCT = "‘’´`"
SUPPORTED_DATASETS = {
    "longbench-hotpotqa": "longbench-hotpotqa",
    "hotpotqa": "longbench-hotpotqa",
    "longbench-triviaqa": "longbench-triviaqa",
    "triviaqa": "longbench-triviaqa",
    "triviaqa-unfiltered-wikipedia": "longbench-triviaqa",
    "longbench-2wiki": "longbench-2wiki",
    "2wiki": "longbench-2wiki",
    "longbench-musique": "longbench-musique",
    "musique": "longbench-musique",
    "longbench-qasper": "longbench-qasper",
    "qasper": "longbench-qasper",
    "longbench-narrativeqa": "longbench-narrativeqa",
    "narrativeqa": "longbench-narrativeqa",
    "conditionalqa": "conditionalqa",
    "dapr-nq-open": "dapr-nq-open",
    "nq-open": "dapr-nq-open",
    "newsqa": "newsqa",
}


def normalize_answer(text: str) -> str:
    return legacy_normalize_answer(text)


def legacy_normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = legacy_normalize_answer(prediction).split()
    gold_tokens = legacy_normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(
        legacy_normalize_answer(prediction) == legacy_normalize_answer(ground_truth)
    )


def clean_prediction(text: str) -> str:
    cleaned = text.strip()

    answer_match = ANSWER_PREFIX_RE.search(cleaned)
    if answer_match:
        cleaned = cleaned[answer_match.end() :]

    cleaned = QUESTION_SPLIT_RE.split(cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    cleaned = lines[0]

    for pattern in TRAILING_SPLIT_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            cleaned = cleaned[: match.start()].strip()

    repeated_match = REPEATED_SENTENCE_RE.match(cleaned)
    if repeated_match:
        cleaned = repeated_match.group("fragment").strip()

    continuation_match = DOUBLE_SPACE_SENTENCE_CONTINUATION_RE.search(cleaned)
    if continuation_match:
        cleaned = cleaned[: continuation_match.start(1) + 1].strip()

    if "..." in cleaned:
        cleaned = cleaned.split("...", 1)[0].strip()

    if " (" in cleaned:
        cleaned = cleaned.split(" (", 1)[0].strip()

    cleaned = cleaned.rstrip(" ,;:([")
    return cleaned


def extract_text_list(value) -> list[str]:
    if value is None:
        return [""]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(extract_text_list(item))
        return flattened
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    return [str(value)]


def extract_prediction_from_record(record: dict, use_cleaner: bool) -> str:
    for key in ("prediction", "pred", "output", "generated_answer", "answer"):
        if key in record:
            candidates = extract_text_list(record[key])
            text = candidates[0] if candidates else ""
            return clean_prediction(text) if use_cleaner else text.strip()

    if "answers" in record:
        candidates = extract_text_list(record["answers"])
        text = candidates[0] if candidates else ""
        return clean_prediction(text) if use_cleaner else text.strip()

    return ""


def extract_ground_truth_from_record(record: dict) -> str:
    if "answer" in record:
        candidates = extract_text_list(record["answer"])
        return candidates[0] if candidates else ""
    if "answers" in record:
        candidates = extract_text_list(record["answers"])
        return candidates[0] if candidates else ""
    return ""


def extract_ground_truths_from_record(record: dict) -> list[str]:
    if "answers" in record:
        candidates = extract_text_list(record["answers"])
        if candidates:
            return candidates
    if "answer" in record:
        candidates = extract_text_list(record["answer"])
        if candidates:
            return candidates
    return [""]


def load_json_records(path: str):
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"{path} is JSON but not a list")
        return data

    records = []
    for line_no, line in enumerate(raw.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"failed to parse JSONL {path} at line {line_no}: {exc}"
            ) from exc
    return records


def load_predictions(prediction_file: str, use_clean_prediction: bool) -> list[str]:
    records = load_json_records(prediction_file)
    return [
        extract_prediction_from_record(record, use_clean_prediction)
        for record in records
    ]


class BaseEvaluator(ABC):
    """Common record-level evaluation loop for dataset-specific scorers."""

    name = "base"

    @abstractmethod
    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        raise NotImplementedError

    def example_ground_truth(self, ground_truth_record: dict) -> str:
        return extract_ground_truth_from_record(ground_truth_record)

    def evaluate_records(
        self,
        predictions: list[str],
        ground_truth_records: list[dict],
        show_examples: int = 10,
    ) -> dict:
        if len(predictions) < len(ground_truth_records):
            ground_truth_records = ground_truth_records[: len(predictions)]
        elif len(predictions) > len(ground_truth_records):
            predictions = predictions[: len(ground_truth_records)]

        em_scores = []
        f1_scores = []
        mismatches = []

        for idx, (prediction, ground_truth_record) in enumerate(
            zip(predictions, ground_truth_records)
        ):
            em, f1 = self.score_prediction(prediction, ground_truth_record)
            em_scores.append(em)
            f1_scores.append(f1)

            if em == 0.0 and len(mismatches) < show_examples:
                mismatches.append(
                    {
                        "id": idx,
                        "prediction": prediction,
                        "ground_truth": self.example_ground_truth(ground_truth_record),
                        "f1": round(f1, 4),
                    }
                )

        count = len(predictions)
        metrics = {
            "count": count,
            "exact_match": (sum(em_scores) / count) if count else 0.0,
            "f1": (sum(f1_scores) / count) if count else 0.0,
            "examples_shown": len(mismatches),
            "mismatches": mismatches,
            "evaluator": self.name,
        }
        return metrics


class LegacyEvaluationEvaluator(BaseEvaluator):
    name = "legacy"

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truth = extract_ground_truth_from_record(ground_truth_record)
        return exact_match(prediction, ground_truth), token_f1(prediction, ground_truth)


class HotpotStyleOfficialEvaluator(BaseEvaluator):
    name = "hotpot_official"

    @staticmethod
    def normalize_answer(text: str) -> str:
        return legacy_normalize_answer(text)

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        if (
            normalized_prediction in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_ground_truth
        ):
            return 0.0
        if (
            normalized_ground_truth in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_ground_truth
        ):
            return 0.0

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        if not prediction_tokens and not ground_truth_tokens:
            return 1.0
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def _exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truths = extract_ground_truths_from_record(ground_truth_record)
        em = max(
            self._exact_match(prediction, ground_truth)
            for ground_truth in ground_truths
        )
        f1 = max(
            self._f1_score(prediction, ground_truth) for ground_truth in ground_truths
        )
        return em, f1


class HotpotOfficialEvaluator(HotpotStyleOfficialEvaluator):
    name = "hotpotqa_official"


class TwoWikiOfficialEvaluator(HotpotStyleOfficialEvaluator):
    name = "2wiki_official"


class MusiqueOfficialEvaluator(HotpotStyleOfficialEvaluator):
    name = "musique_official"


class LongBenchQAEvaluator(BaseEvaluator):
    name = "longbench_qa"

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truths = extract_ground_truths_from_record(ground_truth_record)
        em = max(
            exact_match(prediction, ground_truth) for ground_truth in ground_truths
        )
        f1 = max(token_f1(prediction, ground_truth) for ground_truth in ground_truths)
        return em, f1


class QasperEvaluator(LongBenchQAEvaluator):
    name = "qasper_longbench_qa"


class NarrativeQAEvaluator(LongBenchQAEvaluator):
    name = "narrativeqa_longbench_qa"


def conditionalqa_answer_only_scores(
    predicted_answers: list[str],
    reference_answers: list[str],
) -> tuple[float, float]:
    """Official ConditionalQA permutation scoring without condition labels.

    Ported from ``compute_metrics`` and ``compute_em_f1`` in the official
    ConditionalQA evaluator at commit 77bd295952daf415548b3244db10880d3d55cfe0:
    https://github.com/haitian-sun/ConditionalQA/blob/master/evaluate.py
    """
    if not reference_answers:
        score = float(not predicted_answers)
        return score, score

    num_answers = len(reference_answers)
    padded_predictions = [(answer, []) for answer in predicted_answers]
    if len(padded_predictions) < num_answers:
        padded_predictions.extend([("", [])] * (num_answers - len(padded_predictions)))

    max_em = 0.0
    max_f1 = 0.0
    for ordered_predictions in itertools.permutations(padded_predictions):
        total_em = 0.0
        total_f1 = 0.0
        for (predicted_text, _), reference_text in zip(
            ordered_predictions, reference_answers
        ):
            normalized_prediction = legacy_normalize_answer(predicted_text)
            normalized_reference = legacy_normalize_answer(reference_text)
            total_em += float(normalized_prediction == normalized_reference)
            total_f1 += token_f1(normalized_prediction, normalized_reference)

        max_em = max(max_em, total_em / num_answers)
        max_f1 = max(max_f1, total_f1 / num_answers)

    gamma = math.exp(1.0 - len(padded_predictions) / num_answers)
    return max_em * gamma, max_f1 * gamma


class ConditionalQAAnswerOnlyEvaluator(BaseEvaluator):
    """Official ConditionalQA answer EM/F1 with condition scoring omitted."""

    name = "conditionalqa_official_answer_only"

    @staticmethod
    def _reference_answers(ground_truth_record: dict) -> list[str]:
        answers = ground_truth_record.get("answers")
        if not isinstance(answers, list):
            raise ValueError("ConditionalQA ground truth must contain an answers list")
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError("ConditionalQA answer entries must be strings")
        return answers

    def example_ground_truth(self, ground_truth_record: dict) -> str:
        return json.dumps(
            self._reference_answers(ground_truth_record), ensure_ascii=False
        )

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        predicted_answers = [prediction] if prediction else []
        return conditionalqa_answer_only_scores(
            predicted_answers,
            self._reference_answers(ground_truth_record),
        )


class NQOpenOfficialEvaluator(BaseEvaluator):
    """Official NQ-open exact match with auxiliary token F1.

    Exact-match normalization and max-over-answers scoring are ported from the
    official FiD/DPR evaluator at commit
    fe769f30e3714e22476910ee39ea0054dd7921de:
    https://github.com/facebookresearch/FiD/blob/fe769f30e3714e22476910ee39ea0054dd7921de/src/evaluation.py

    The upstream evaluator reports exact match only. Token F1 is retained here
    as an explicitly auxiliary diagnostic and is not an official NQ-open score.
    """

    name = "nq_open_official_em_aux_token_f1"

    @staticmethod
    def normalize_answer(text: str) -> str:
        def remove_articles(value: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", value)

        def white_space_fix(value: str) -> str:
            return " ".join(value.split())

        def remove_punc(value: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in value if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(text.lower())))

    @staticmethod
    def _reference_answers(ground_truth_record: dict) -> list[str]:
        answers = ground_truth_record.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(
                "NQ-open ground truth must contain a non-empty answers list"
            )
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError("NQ-open answer entries must be strings")
        return answers

    def _exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )

    def _auxiliary_token_f1(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()

        if not prediction_tokens and not ground_truth_tokens:
            return 1.0
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def example_ground_truth(self, ground_truth_record: dict) -> str:
        return json.dumps(
            self._reference_answers(ground_truth_record), ensure_ascii=False
        )

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truths = self._reference_answers(ground_truth_record)
        exact_match_score = max(
            self._exact_match(prediction, ground_truth)
            for ground_truth in ground_truths
        )
        auxiliary_token_f1 = max(
            self._auxiliary_token_f1(prediction, ground_truth)
            for ground_truth in ground_truths
        )
        return exact_match_score, auxiliary_token_f1


class NewsQAOfficialEvaluator(BaseEvaluator):
    """NewsQA EM/F1 using the official SQuAD v1.1 scoring rules.

    The NewsQA paper specifies the official SQuAD evaluator:
    https://aclanthology.org/W17-2623/

    This implementation ports:
    https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
    """

    name = "newsqa_official_squad_v1_1"

    @staticmethod
    def normalize_answer(text: str) -> str:
        def remove_articles(value: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", value)

        def white_space_fix(value: str) -> str:
            return " ".join(value.split())

        def remove_punc(value: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in value if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(text.lower())))

    @staticmethod
    def _reference_answers(ground_truth_record: dict) -> list[str]:
        answers = ground_truth_record.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(
                "NewsQA ground truth must contain a non-empty answers list"
            )
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError("NewsQA answer entries must be strings")
        return answers

    def _exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def example_ground_truth(self, ground_truth_record: dict) -> str:
        return self._reference_answers(ground_truth_record)[0]

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truths = self._reference_answers(ground_truth_record)
        exact_match_score = max(
            self._exact_match(prediction, ground_truth)
            for ground_truth in ground_truths
        )
        f1_score = max(
            self._f1_score(prediction, ground_truth) for ground_truth in ground_truths
        )
        return exact_match_score, f1_score


class TriviaQAOfficialEvaluator(BaseEvaluator):
    name = "triviaqa_official"

    @staticmethod
    def normalize_answer(text: str) -> str:
        def remove_articles(value: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", value)

        def white_space_fix(value: str) -> str:
            return " ".join(value.split())

        def handle_punc(value: str) -> str:
            exclude = set(string.punctuation + TRIVIA_EXTRA_PUNCT)
            return "".join(ch if ch not in exclude else " " for ch in value)

        def lower(value: str) -> str:
            return value.lower()

        return white_space_fix(
            remove_articles(handle_punc(lower(text.replace("_", " "))))
        ).strip()

    def _exact_match_score(self, prediction: str, ground_truth: str) -> float:
        return float(
            self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
        )

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        if not prediction_tokens and not ground_truth_tokens:
            return 1.0
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def _get_ground_truths(self, ground_truth_record: dict) -> list[str]:
        if "answers" in ground_truth_record:
            candidates = extract_text_list(ground_truth_record["answers"])
            if candidates:
                return candidates

        answer_value = ground_truth_record.get("answer")
        if isinstance(answer_value, dict):
            aliases = extract_text_list(answer_value.get("NormalizedAliases"))
            human_answers = [
                self.normalize_answer(candidate)
                for candidate in extract_text_list(answer_value.get("HumanAnswers"))
            ]
            return aliases + human_answers

        candidates = extract_text_list(answer_value)
        if not candidates and "answers" in ground_truth_record:
            candidates = extract_text_list(ground_truth_record["answers"])
        return candidates or [""]

    def example_ground_truth(self, ground_truth_record: dict) -> str:
        candidates = self._get_ground_truths(ground_truth_record)
        return candidates[0] if candidates else ""

    def score_prediction(
        self, prediction: str, ground_truth_record: dict
    ) -> tuple[float, float]:
        ground_truths = self._get_ground_truths(ground_truth_record)
        em = max(
            self._exact_match_score(prediction, ground_truth)
            for ground_truth in ground_truths
        )
        f1 = max(
            self._f1_score(prediction, ground_truth) for ground_truth in ground_truths
        )
        return em, f1


def normalize_dataset_name(dataset: Optional[str]) -> Optional[str]:
    if dataset is None:
        return None
    normalized = dataset.lower()
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(set(SUPPORTED_DATASETS.values())))
        raise ValueError(
            f"unsupported dataset '{dataset}'. Supported datasets: {supported}"
        )
    return SUPPORTED_DATASETS[normalized]


def build_evaluator(dataset: Optional[str]) -> BaseEvaluator:
    normalized_dataset = normalize_dataset_name(dataset)
    if normalized_dataset is None:
        return LegacyEvaluationEvaluator()
    if normalized_dataset == "longbench-hotpotqa":
        return HotpotOfficialEvaluator()
    if normalized_dataset == "longbench-triviaqa":
        return TriviaQAOfficialEvaluator()
    if normalized_dataset == "longbench-2wiki":
        return TwoWikiOfficialEvaluator()
    if normalized_dataset == "longbench-musique":
        return MusiqueOfficialEvaluator()
    if normalized_dataset == "longbench-qasper":
        return QasperEvaluator()
    if normalized_dataset == "longbench-narrativeqa":
        return NarrativeQAEvaluator()
    if normalized_dataset == "conditionalqa":
        return ConditionalQAAnswerOnlyEvaluator()
    if normalized_dataset == "dapr-nq-open":
        return NQOpenOfficialEvaluator()
    if normalized_dataset == "newsqa":
        return NewsQAOfficialEvaluator()
    raise ValueError(f"no evaluator configured for dataset '{dataset}'")


def evaluate(
    prediction_file: str,
    ground_truth_file: str,
    dataset: Optional[str] = None,
    use_cleaner: bool = True,
    show_examples: int = 10,
    use_clean_prediction: Optional[bool] = None,
):
    if use_clean_prediction is not None:
        use_cleaner = use_clean_prediction

    predictions = load_predictions(prediction_file, use_clean_prediction=use_cleaner)
    ground_truth_records = load_json_records(ground_truth_file)
    evaluator = build_evaluator(dataset)
    metrics = evaluator.evaluate_records(
        predictions=predictions,
        ground_truth_records=ground_truth_records,
        show_examples=show_examples,
    )
    metrics["dataset"] = (
        normalize_dataset_name(dataset) if dataset is not None else "legacy"
    )
    metrics["use_cleaner"] = use_cleaner

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics
