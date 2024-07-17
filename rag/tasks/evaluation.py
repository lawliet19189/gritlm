import logging
import string
from collections import Counter
from typing import Callable

import regex

logger = logging.getLogger(__name__)


# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]) -> float:
    """Check if the normalized prediction is equal to the normalized ground truth."""
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def match(
    prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]
) -> float:
    """Check if the normalized prediction is a substring of the normalized ground truth."""
    return float(normalize_fn(ground_truth) in normalize_fn(prediction))


def f1(prediction: str, ground_truth: str, normalize_fn) -> float:
    """Compute the F1 score between a prediction and a ground truth."""
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge_wrapper(prediction: str, ground_truth: str) -> float:
    """Compute the Rouge score between a prediction and a ground truth."""
    from rouge import Rouge  # pylint: disable=import-outside-toplevel

    rouge = Rouge()
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        logger.warning("Rouge failed (%s), returning 0", e)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Rouge failed (%s), returning 0", e)
        return 0.0, 0.0, 0.0


def f1_score(
    prediction: str,
    ground_truths: list[str],
    normalize_fn: Callable[[str], str] = lambda x: x,
) -> float:
    """Compute the F1 score between a prediction and multiple ground truth using F1 score."""
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(
    prediction: str,
    ground_truths: list[str],
    normalize_fn: Callable[[str], str] = lambda x: x,
) -> float:
    """Compute the exact match score between a prediction and multiple ground truth using exact match."""
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def match_score(
    prediction: str,
    ground_truths: list[str],
    normalize_fn: Callable[[str], str] = lambda x: x,
) -> float:
    """Compute the match score between a prediction and multiple ground using substring match."""
    return max([match(prediction, gt, normalize_fn) for gt in ground_truths])


def rouge_score(
    prediction: str, ground_truths: list[str]
) -> tuple[float, float, float]:
    """Compute the Rouge score between a prediction and multiple ground truth."""
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel
