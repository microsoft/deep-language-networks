import logging
from dataclasses import dataclass
from typing import Any, List
import numpy as np

from abc import ABC, abstractmethod
from ops import forward_evaluate


def prepare_prompts_scoring_args(
    inputs: np.ndarray, outputs: np.ndarray, prompts: np.ndarray
):
    """
    Args:
        inputs: (batch_size,)
        outputs: (batch_size, 1) or (batch_size, num_outputs)
        prompts: (num_prompts,)

    Returns
        (batch_size, num_outputs, num_prompts)
    """

    # add a dimension in case there is only 1 output
    if outputs.ndim == 1:
        outputs = outputs[:, None]

    evals = []
    for i in range(inputs.shape[0]):
        for j in range(outputs.shape[1]):
            for k in range(prompts.shape[0]):
                evals.append(
                    (
                        inputs[i],
                        outputs[i, j],
                        prompts[k],
                    )
                )
    return list(zip(*evals))


def prepare_inputs_scoring_args(inputs: np.ndarray, outputs: np.ndarray, prompt: str):
    """
    Args:
        inputs: (batch_size, num_inputs)
        outputs: (batch_size, 1) or (batch_size, num_outputs)
        prompts: (num_prompts,)

    Returns
        (batch_size, num_inputs, num_outputs)
    """

    # add a dimension in case there is only 1 output
    if outputs.ndim == 1:
        outputs = outputs[:, None]

    evals = []
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(outputs.shape[1]):
                evals.append(
                    (
                        inputs[i, j],
                        outputs[k],
                        prompt,
                    )
                )
    return list(zip(*evals))


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
    def __init__(self, encoder=None):
        if encoder is None:
            import tiktoken
            from ops import forward_interpreter

            encoder = tiktoken.encoding_for_model(forward_interpreter.engine)
        self.encoder = encoder

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

        unique_contexts = list(set(contexts))
        context_to_position = {context: i for i, context in enumerate(unique_contexts)}
        to_eval = [f"{context}\n" for context in unique_contexts]

        logging.debug("# Scoring requests = {}".format(len(contexts)))
        logging.debug("# Scoring unique requests = {}".format(len(unique_contexts)))
        eval_results = forward_evaluate(
            to_eval,
            async_generation=True,
            **eval_kwargs,
        )

        top_logprobs = []
        for context in contexts:
            position = context_to_position[context]
            context_top_logprobs = eval_results[position][1][0]
            top_logprobs.append(dict(context_top_logprobs))

        output_logprobs = []
        output_distribs = []
        for context, target, context_top_logprobs in zip(contexts, targets, top_logprobs):
            position = context_to_position[context]

            # make this fixed
            if context_top_logprobs:
                min_prob = np.exp(np.min(list(context_top_logprobs.values())))
            else:
                min_prob = 1e-6

            output_classes_scores = np.asarray([min_prob for _ in output_classes])
            # accumulate probability mass for each class verbalizer
            # the class verbalizer can be either " a" or "a" (with or without space)
            for i in range(len(output_classes)):
                verbalizers = output_classes.verbalizers(i)
                verbalizers.extend([f" {v}" for v in verbalizers])
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
        logging.debug("# Scoring requests = {}".format(len(contexts)))

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
        unique_eval_results = forward_evaluate(
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
            num_tokens_prompt = len(self.encoder.encode(context))
            target_log_probs = token_log_probs[num_tokens_prompt:]
            context_log_probs = token_log_probs[1:num_tokens_prompt]
            output_logprobs.append(sum(target_log_probs) / (len(target_log_probs) + 1e-5))
            context_logprobs.append(sum(context_log_probs) / (len(context_log_probs) + 1e-5))
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))


class Scorer(ABC):
    def __init__(self):
        # dependency injection
        self.base_layer: NetworkNode = None

    def register_base_layer(self, base_layer):
        self.base_layer = base_layer

    @abstractmethod
    def score_prompts(self, candidate_prompts, y, weights):
        pass

    @abstractmethod
    def score_inputs(self, candidate_inputs, y):
        pass


class LogProbsScorer(Scorer):
    def score_prompts(self, candidate_prompts, y, y_weights):
        num_samples = candidate_prompts.shape[-1]
        args = prepare_prompts_scoring_args(
            self.base_layer.inputs_cache, y, candidate_prompts
        )
        # build up a set of score requests
        requests = []
        for input, target, prompt in zip(*args):
            input = self.base_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))
        prompt_logprobs = LogProbsScore().score_requests(
            requests,
            self.base_layer.output_classes,
            agg="sum",
        ).logp_targets
        prompt_logprobs = prompt_logprobs.reshape(
            self.base_layer.inputs_cache.shape[0], y.shape[1], num_samples
        )
        prompt_scores = (y_weights[:, :, None] * prompt_logprobs).sum(1).mean(0)
        return prompt_scores

    def score_inputs(self, candidate_inputs, y):
        args = prepare_inputs_scoring_args(candidate_inputs, y, self.base_layer.weight)
        requests = []
        for input, target, prompt in zip(*args):
            input = self.base_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))
        input_scores = LogProbsScore().score_requests(
            requests,
            self.base_layer.output_classes,
            agg="sum",
        ).logp_targets
        input_scores = input_scores.reshape(
            self.base_layer.inputs_cache.shape[0],
            candidate_inputs.shape[1],
        )
        input_scores = np.exp(input_scores) / np.exp(input_scores).sum(
            1, keepdims=True
        )
        return input_scores


class AccuracyScorer(Scorer):
    def score_prompts(self, candidate_prompts, y, y_weights):
        from postprocessing import postprocess_prediction
        
        num_samples = candidate_prompts.shape[-1]
        args = prepare_prompts_scoring_args(
            self.base_layer.inputs_cache, y, candidate_prompts
        )
        # build up a set of score requests
        requests = []
        for input, _, prompt in zip(*args):
            requests.append(
                self.base_layer.instantiate_template(input=input, prompt=prompt)
            )
        outputs = forward_evaluate(
            requests,
            stop=["\n\n"],
            temperature=0.0,
            max_tokens=10,
        )
        targets = np.array([postprocess_prediction(t) for t in targets])
        outputs = np.array([postprocess_prediction(yi) for yi in y])
        accuracy = np.array([t == y for t, y in zip(targets, outputs)])
        prompt_weights = accuracy.reshape(
            self.base_layer.inputs_cache.shape[0], y.shape[1], num_samples
        )
        prompt_scores = (y_weights[:, :, None] * prompt_weights).sum(1).mean(0)
        return prompt_scores

    def score_inputs(self, candidate_inputs, y):
        raise NotImplementedError("Cannot score inputs with accuracy!")
