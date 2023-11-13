import logging
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from dln.operator import LLM


@dataclass
class ScoreRequest:
    context: str
    target: str
    payload: Any = None


@dataclass
class OutputClasses:
    protos: List[str]

    def __iter__(self):
        return iter(self.protos)

    def __len__(self):
        return len(self.protos)

    def verbalizers(self, i):
        return self.protos[i].split("|")

    def prototype(self, i):
        return self.protos[i].split("|")[0]


@dataclass
class LogProbs:
    logp_targets: np.ndarray
    distribution: np.ndarray


class LogProbsScore:

    def __init__(self, forward_evaluate: LLM):
        self.forward_evaluate = forward_evaluate
        self.cache = {}

    def score_requests(self, requests, output_classes=None, agg="max") -> LogProbs:
        # create the batched inputs for the model
        if output_classes is not None:
            return self._forward_logprobs_score_api_with_classes(
                [b.context for b in requests],
                [b.target for b in requests],
                output_classes,
                agg=agg,
            )
        return self._forward_logprobs_score_api(
            [b.context for b in requests],
            [b.target for b in requests],
        )

    def _forward_logprobs_score_api_with_classes(
        self, contexts, targets, output_classes, agg="max"
    ) -> LogProbs:
        eval_kwargs = {
            "temperature": 0.,
            "max_tokens": 1,
            "echo": False,
            "return_logprobs": True,
            "raw_logprobs": False,
            "top_logprobs": 100,
        }
        
        output_logprobs = []
        output_distribs = []

        to_eval = []
        for context in contexts:
            if context not in self.cache:
                context_ = f"{context}\n"
                to_eval.append(context_)

        print("# Scoring requests = {}".format(len(contexts)))
        print("# Scoring non cached requests = {}".format(len(to_eval)))

        partial_results = self.forward_evaluate(
            to_eval,
            async_generation=True,
            **eval_kwargs,
        )
        for context, result in zip(to_eval, partial_results):
            assert context not in self.cache
            self.cache[context.strip()] = result

        eval_results = []
        for context in contexts:
            eval_results.append(self.cache[context])

        top_logprobs = []
        for context, result in zip(contexts, eval_results):
            context_top_logprobs = result[1][0]
            top_logprobs.append(dict(context_top_logprobs))

        output_logprobs = []
        output_distribs = []
        for context, target, context_top_logprobs in zip(contexts, targets, top_logprobs):
            # make this fixed
            if context_top_logprobs:
                min_prob = np.exp(np.min(list(context_top_logprobs.values())))
            else:
                min_prob = 1e-6

            output_classes_scores = np.asarray([min_prob for _ in output_classes])
            # accumulate probability mass for each class verbalizer
            # the class verbalizer can be either " a" or "a" (with or without space)
            extenders = [
                " ",  # gpt case, regular space
                "▁",  # llama case, this is not an underscore ord("_") -> 95, but a special character ord("▁") -> 9601
            ]
            for i in range(len(output_classes)):
                verbalizers = output_classes.verbalizers(i)
                verbalizers.extend([f"{e}{v}" for v in verbalizers for e in extenders])
                verbalizers = set(verbalizers)
                verbalizers_scores = [0.]
                for verbalizer in verbalizers:
                    if verbalizer in context_top_logprobs:
                        prob_orig = np.exp(context_top_logprobs[verbalizer])
                    else:
                        prob_orig = min_prob
                    verbalizers_scores.append(prob_orig)
                if agg == "max":
                    output_classes_scores[i] += np.max(verbalizers_scores)
                else:
                    output_classes_scores[i] += np.sum(verbalizers_scores)
            output_class_index = [i for i, output_class in enumerate(output_classes) if target in output_class.split("|")]
            assert (
                len(output_class_index) == 1
            ), "The target shouldn't appear in two output classes! {}".format(target)
            # accuracy here
            output_classes_scores = output_classes_scores / output_classes_scores.sum()
            output_logprobs.append(np.log(output_classes_scores[output_class_index[0]]))
            output_distribs.append(output_classes_scores)
        return LogProbs(np.asarray(output_logprobs), np.asarray(output_distribs))

    def _forward_logprobs_score_api(self, contexts, targets) -> LogProbs:
        logging.info("# Scoring requests = {}".format(len(contexts)))
        eval_kwargs = {
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "return_logprobs": True,
            "raw_logprobs": True,
        }

        eval_batch = []
        for context, target in zip(contexts, targets):
            to_eval = f"{context}\n{target}"
            eval_batch.append(to_eval)

        # there might be doubles in the eval_batch, so we need to
        # only perform unique evals
        unique_keys = list(set(eval_batch))
        unique_keys_to_positions = {key: i for i, key in enumerate(unique_keys)}
        unique_eval_results = self.forward_evaluate(
            unique_keys,
            async_generation=True,
            **eval_kwargs,
        )
        # get the results in the same order as the eval_batch
        eval_results = []
        for eval_key in eval_batch:
            eval_results.append(unique_eval_results[unique_keys_to_positions[eval_key]])
        # get the nll results
        log_probs = [eval_result[1] for eval_result in eval_results]
        # get the logprobs results
        output_logprobs = []
        context_logprobs = []
        for context, token_log_probs in zip(contexts, log_probs):
            num_tokens_prompt = len(self.forward_evaluate.encode(context))
            target_log_probs = token_log_probs[num_tokens_prompt:]
            context_log_probs = token_log_probs[1:num_tokens_prompt]
            output_logprobs.append(sum(target_log_probs) / (len(target_log_probs) + 1e-5))
            context_logprobs.append(sum(context_log_probs) / (len(context_log_probs) + 1e-5))
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))
