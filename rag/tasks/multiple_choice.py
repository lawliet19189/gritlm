# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of the ATLAS repo: https://github.com/facebookresearch/atlas.

import copy
import itertools
import string
from typing import Any, Optional

from rag.prompt_choices import PromptType
from rag.tasks.evaluation import (
    exact_match_score,
    f1_score,
    match_score,
    normalize_answer,
)

from rag.tasks.base import BaseTask


def _get_permutation_orderings(N, permutations_type):
    li = list(range(N))
    if permutations_type == "cyclic":
        orderings = [li[N - i :] + li[: N - i] for i in range(N)]
    elif permutations_type == "all":
        orderings = list(itertools.permutations(li))
    else:
        orderings = [li]
    return orderings


class Task(BaseTask):
    metrics = ["debiased_accuracy", "accuracy", "eval_loss"]

    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = string.ascii_uppercase[: opt.multiple_choice_num_options]

    @staticmethod
    def get_prompt_format(prompt_type: PromptType) -> str:
        """Returns the prompt format string for the given prompt type"""
        if prompt_type == PromptType.NO_RETRIEVAL:
            return "<|user|>\n{query}\n{options}\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT:
            return "<|embed|>\n{query}\n{options}\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_NO_EMBED:
            return "<|user|>\n{query}\n{options}\n\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_DOC:
            return "<|embed|>\n{title} {text}\n<|user|>\n{query}\n{options}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.FULL_FORMAT_NO_EMBED_DOC:
            return "<|user|>\n{title} {text}\n\n{query}\n{options}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_QUERY:
            return "\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_DOC:
            return "\n<|user|>\n{query}\n{options}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_DOC_QUERY:
            return "\n<|user|>\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
        elif prompt_type == PromptType.CACHE_FORMAT_QUERY_DOC:
            return "\n<|user|>\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
        else:
            raise ValueError(f"Prompt type {prompt_type} not recognized")

    @staticmethod
    def _format_choices(choices: dict[str, Any]) -> str:
        """Formats the choices as a string"""
        return "\n".join(
            [f"{letter}. {choice}" for letter, choice in choices.items()]
        )

    def get_formatted_task_prompt(
        self,
        text_and_metadata: dict[str, Any],  # TODO: support list of passages
        question: str,
        prompt_type: PromptType,
        choices: Optional[dict[str, Any]] = None,
    ) -> str:
        """Returns the formatted task prompt"""
        unformatted_prompt = Task.get_prompt_format(prompt_type)
        return unformatted_prompt.format(
            query=question,
            **text_and_metadata,
            options=Task._format_choices(choices),
        )

    @staticmethod
    def get_permutations(example, permutations_type):
        """clones example according to permutations_type (either "none", 'cyclic' or 'full'"""
        options, answer = example["options"], example["answer"]
        uid = example["question"] + " ".join(options.values())

        choice_keys = list(sorted(options.keys()))
        choice_values = [options[l] for l in choice_keys]
        orderings = _get_permutation_orderings(
            len(choice_keys), permutations_type
        )

        permuted_examples = []
        for ordering in orderings:
            permuted_options = {
                l: choice_values[o] for l, o in zip(choice_keys, ordering)
            }
            permuted_answer = [
                k
                for k, ans in permuted_options.items()
                if ans == options[answer]
            ][0]

            permed_example = copy.deepcopy(example)
            permed_example["options"] = permuted_options
            permed_example["answer"] = permuted_answer
            permed_example["is_original"] = (
                permuted_options == example["options"]
            )
            permed_example["uid"] = uid
            permuted_examples.append(permed_example)

        return permuted_examples

    @staticmethod
    def data_iterator(*args, **kwargs):
        # wrap base data iterator in the case of permuting examples
        super_iterator = super(Task, Task).data_iterator(*args, **kwargs)
        perms_type = (
            kwargs["opt"].multiple_choice_eval_permutations
            if kwargs.get("is_eval", False)
            else kwargs["opt"].multiple_choice_train_permutations
        )
        for example in super_iterator:
            for permed_item in Task.get_permutations(example, perms_type):
                yield permed_item

    def evaluation(self, prediction, ground_truths):
        """Computes exact match, match and F1 score between prediction and multiple ground truth."""
        sample_metrics = {
            "exact_match": exact_match_score(
                prediction, ground_truths, normalize_answer
            ),
            "match": match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics

    @staticmethod
    def _answer_formatter(example) -> dict[str, Any]:
        for option_letter, option_desc in example["options"].items():
            # 'answer' contains only the letter.
            if example["answer"] == option_letter:
                example["answer"] = f"{option_letter}. {option_desc}."
                break
        return example

    def process(self, example, *args, **kwargs) -> dict[str, Any]:
        assert (
            "question" in example
        ), "multiple_choice task requires a `question` field string to be defined"
        assert (
            "options" in example
        ), "multiple_choice task requires a `options` field string to be defined"

        example = Task._answer_formatter(example)

        return {
            "query": example["question"],
            "options": example["options"],
            "choices": self.choices,
            "passages": [{"title": "", "text": ""}],
            "answers": [example["answer"]],
            "metadata": example,
        }
