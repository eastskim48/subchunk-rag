"""Custom BioASQ scoring with user-specified whitespace/comma List parsing.

List PRF follows the pinned BioASQ evaluator's matching/counting operations:
Evaluation-Measures 8c8e689068a41c1370fec55260e5fbfe3a6a6016,
QuestionAnswerEvaluator.calculatePRFforListQuestions and
ExactAnswer.containsAnswerSynonym (Apache-2.0).
"""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

from .acc_metric import NewsQAOfficialEvaluator


def evaluate_saved_answers(prediction_file, ground_truth_file, query_file):
    """Use the existing custom BioASQ protocol through the native eval entrypoint."""

    def rows(path):
        with Path(path).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    predictions = rows(prediction_file)
    gold = rows(ground_truth_file)
    queries = rows(query_file)
    if len(predictions) != len(queries) or len(gold) != len(queries):
        raise ValueError("Incomplete BioASQ prediction/gold records")
    for query, prediction in zip(queries, predictions, strict=True):
        if query["id"] != prediction["id"] or query["query"] != prediction["question"]:
            raise ValueError("Question ID/order/text mismatch")
    result = evaluate_questions(gold, predictions)
    output = Path(prediction_file).parent
    for filename, value in (
        ("scores.json", result),
        ("results.json", {k: v for k, v in result.items() if k != "records"}),
    ):
        path = output / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    print("BIOASQ_SCORES " + json.dumps(result["overall"]), flush=True)
    print(
        json.dumps(
            dict(
                count=len(queries),
                f1=result["overall"]["f1"],
                exact_match=result["overall"]["em"],
                dataset="bioasq",
                evaluator="existing_bioasq_metrics",
            )
        ),
        flush=True,
    )
    Path(prediction_file).with_suffix(".bioasq_scores.json").write_bytes(
        (output / "scores.json").read_bytes()
    )
    return result


def _remove_matching_group(groups, candidate):
    for index, group in enumerate(groups):
        if any(value in group for value in candidate):
            groups.pop(index)
            return True
    return False


def parse_list_answer(text):
    """Remove whitespace, then split on literal commas; retain order and empties."""
    if not isinstance(text, str):
        raise ValueError("prediction must be the raw generated answer string")
    return "".join(text.split()).split(",")


def list_item_scores(question, prediction):
    """Custom comma parsing/whole-list EM plus BioASQ item F1 counting."""
    items = parse_list_answer(prediction["prediction"])
    gold = question["exact_answer"]
    if (
        not isinstance(gold, list)
        or not gold
        or not all(
            isinstance(group, list) and group and all(isinstance(x, str) for x in group)
            for group in gold
        )
    ):
        raise ValueError("List exact_answer must contain nonempty synonym groups")
    # Retain official lowercase comparison; whitespace removal is user-defined.
    normalized_gold = [
        ["".join(text.split()).lower() for text in group] for group in gold
    ]
    normalized_items = [text.lower() for text in items]
    em = float(
        len(normalized_items) == len(normalized_gold)
        and all(item in group for item, group in zip(normalized_items, normalized_gold))
    )
    remaining_gold = deepcopy(normalized_gold)
    response = [[text] for text in normalized_items]
    tp = fp = fn = 0
    if response:
        for candidate in response:
            if _remove_matching_group(remaining_gold, candidate):
                tp += 1
            else:
                fp += 1
        for group in remaining_gold:
            if not _remove_matching_group(response, group):
                fn += 1
    else:
        fn = len(gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "em": em,
        "metric": "custom_comma_list_item_f1_ordered_em",
        "parsed_items": items,
        "normalized_items": normalized_items,
    }


def single_answer_scores(question, prediction):
    references = question["exact_answer"]
    if question["type"] == "yesno":
        references = [references]
    else:
        # Factoid groups are alternative names, unlike List's required items.
        references = [
            text
            for entry in references
            for text in (entry if isinstance(entry, list) else [entry])
        ]
    text = prediction["prediction"]
    if not isinstance(text, str):
        raise ValueError("prediction must retain the generated answer string")
    em, f1 = NewsQAOfficialEvaluator().score_prediction(text, {"answers": references})
    return {"em": em, "f1": f1, "metric": "custom_max_reference_token_f1_em"}


DEFAULT_SCORERS = {
    "factoid": single_answer_scores,
    "yesno": single_answer_scores,
    "list": list_item_scores,
}


def evaluate_questions(questions, predictions, *, scorer_overrides=None):
    """Macro-average question scores; allow later replacement by question type.

    Each scorer receives the complete original question and prediction records.
    Summary questions alone are excluded. Missing/duplicate/extra predictions fail
    rather than silently changing the evaluation denominator. Raw outputs and gold
    annotations are included in the returned records for later independent scoring.
    """
    unknown = {q["type"] for q in questions} - {*DEFAULT_SCORERS, "summary"}
    if unknown:
        raise ValueError(f"Unknown question types: {unknown}")
    ids = [q["id"] for q in questions]
    pred_ids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or len(pred_ids) != len(set(pred_ids)):
        raise ValueError("Duplicate question/prediction IDs")
    selected = [q for q in questions if q["type"] != "summary"]
    expected = {q["id"] for q in selected}
    summary_ids = {q["id"] for q in questions if q["type"] == "summary"}
    if expected - set(pred_ids) or set(pred_ids) - expected - summary_ids:
        raise ValueError("Prediction IDs do not match the non-summary questions")
    if not selected:
        raise ValueError("No non-summary questions")
    scorers = dict(DEFAULT_SCORERS)
    if scorer_overrides:
        if set(scorer_overrides) - set(scorers):
            raise ValueError("Scorer override must name a supported question type")
        scorers.update(scorer_overrides)
    by_id = {p["id"]: p for p in predictions}
    rows = []
    for question in selected:
        prediction = by_id[question["id"]]
        scores = scorers[question["type"]](deepcopy(question), deepcopy(prediction))
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "gold_record": deepcopy(question),
                "prediction_record": deepcopy(prediction),
                "scores": scores,
            }
        )

    def aggregate(group):
        em_values = [r["scores"].get("em") for r in group]
        return {
            "questions": len(group),
            "f1": mean(r["scores"]["f1"] for r in group),
            # Never report a partial-denominator overall EM as a full-set EM.
            "em": mean(em_values) if all(x is not None for x in em_values) else None,
        }

    return {
        "protocol": "custom_bioasq_non_summary_mixed_f1_comma_ordered_em_v1",
        "excluded_summary": sum(q["type"] == "summary" for q in questions),
        "overall": aggregate(rows),
        "by_type": {
            kind: aggregate([r for r in rows if r["type"] == kind])
            for kind in Counter(r["type"] for r in rows)
        },
        "records": rows,
    }
