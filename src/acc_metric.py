import json
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
    "longbench-2wiki": "longbench-2wiki",
    "2wiki": "longbench-2wiki",
    "longbench-musique": "longbench-musique",
    "musique": "longbench-musique",
    "longbench-qasper": "longbench-qasper",
    "qasper": "longbench-qasper",
    "longbench-narrativeqa": "longbench-narrativeqa",
    "narrativeqa": "longbench-narrativeqa",
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
    for line_no, line in enumerate(raw.splitlines(), start=1):
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


def load_ground_truths(ground_truth_file: str) -> list[str]:
    records = load_json_records(ground_truth_file)
    return [extract_ground_truth_from_record(record) for record in records]


class BaseEvaluator(ABC):
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
