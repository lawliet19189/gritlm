import random
from typing import Any, Optional

from rag.prompt_choices import PromptType
from rag.tasks.evaluation import (
    exact_match_score,
    match_score,
    f1_score,
    normalize_answer,
)
from rag.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["exact_match", "match", "f1", "eval_loss"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def get_prompt_format(prompt_type: PromptType) -> str:
        """Returns the prompt format string for the given prompt type"""
        if prompt_type == PromptType.NO_RETRIEVAL:
            return "<|user|>\n{query}\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT:
            return "<|embed|>\n{query}\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_NO_EMBED:
            return "<|user|>\n{query}\n\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_DOC:
            return "<|embed|>\n{title} {text}\n<|user|>\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_NO_EMBED_DOC:
            return "<|user|>\n{title} {text}\n\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_QUERY:
            return "\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_DOC:
            return "\n<|user|>\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_DOC_QUERY:
            return "\n<|user|>\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_QUERY_DOC:
            return "\n<|user|>\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        else:
            raise ValueError(f"Prompt type {prompt_type} not recognized")

    def get_formatted_task_prompt(
        self,
        text_and_metadata: dict[str, Any],  # TODO: support list of passages
        question: str,
        prompt_type: PromptType,
        choices: Optional[dict[str, Any]] = None,  # type: ignore
    ) -> str:
        """Returns the formatted task prompt"""
        unformatted_prompt = Task.get_prompt_format(prompt_type)
        return unformatted_prompt.format(
            query=question,
            **text_and_metadata,
        )

    def evaluation(
        self, prediction: str, ground_truths: list[str]
    ) -> dict[str, float]:
        """Computes exact match, match and F1 score between prediction and multiple ground truth."""
        sample_metrics = {
            "exact_match": exact_match_score(
                prediction, ground_truths, normalize_answer
            ),
            "match": match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
