import logging

from dataclasses import dataclass
from typing import List
import numpy as np

from abc import ABC, abstractmethod
from ops import GPT, ScoreRequest
from network import NetworkNode
from layers import LanguageLayer


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
    if inputs.ndim == 1:
        inputs = inputs[:, None]

    evals = []
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(outputs.shape[1]):
                evals.append(
                    (
                        inputs[i, j],
                        outputs[i, k],
                        prompt,
                    )
                )
    return list(zip(*evals))


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


class Scorer(ABC):
    def __init__(self):
        # dependency injection
        self._base_layer: NetworkNode = None

    def register_base_layer(self, base_layer):
        self._base_layer = base_layer

    @property
    def scoring_lm(self) -> GPT:
        return self.base_layer.scoring_lm

    @property
    def base_layer(self) -> LanguageLayer:
        if not self._base_layer:
            raise ValueError("Base layer not set, did you register this scorer?")
        return self._base_layer

    @abstractmethod
    def score_prompts(self, candidate_prompts, y, weights):
        pass

    @abstractmethod
    def score_inputs(self, candidate_inputs, y):
        pass


class LogProbsScorer(Scorer):
    def score_prompts(
        self, candidate_prompts: np.array, y: np.array, y_weights: np.ndarray
    ):
        (num_samples,) = candidate_prompts.shape
        batch_size, num_targets = y.shape

        # build up a set of score requests
        requests = []
        args = prepare_prompts_scoring_args(
            self.base_layer.inputs_cache, y, candidate_prompts
        )
        for input, target, prompt in zip(*args):
            input = self.base_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))

        lps = self.scoring_lm.compute_log_p(
            requests,
            self.base_layer.output_classes,
        ).logp_targets.reshape(batch_size, num_targets, num_samples)

        prompt_scores = (y_weights[:, :, None] * lps).sum(1).mean(0)
        return prompt_scores

    def score_inputs(self, candidate_inputs, y, candidate_inputs_logps=None):
        args = prepare_inputs_scoring_args(candidate_inputs, y, self.base_layer.weight)

        batch_size, num_samples = candidate_inputs.shape
        requests = []
        for input, target, prompt in zip(*args):
            input = self.base_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))

        lps = self.scoring_lm.compute_log_p(
            requests,
            self.base_layer.output_classes,
            agg="sum",
        ).logp_targets.reshape(
            batch_size,
            num_samples,
        )

        input_scores = np.exp(lps) / np.exp(lps).sum(1, keepdims=True)
        return input_scores


class VIScorer(LogProbsScorer):
    def score_inputs(self, candidate_inputs, y, candidate_inputs_logps=None):
        args = prepare_inputs_scoring_args(candidate_inputs, y, self.base_layer.weight)
        requests = []

        batch_size, num_samples = candidate_inputs.shape
        if y.ndim == 1:
            y = y[:, None]

        num_output_targets = y.shape[1]
        assert num_output_targets == 1

        for input, target, prompt in zip(*args):
            input = self.base_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))

        lps = self.scoring_lm.compute_log_p(
            requests,
            self.base_layer.output_classes,
            agg="sum",
        ).logp_targets.reshape(
            batch_size,
            num_samples,
            num_output_targets,
        ).squeeze()

        parent_layer = self.base_layer.input_nodes[0]
        args = prepare_inputs_scoring_args(
            parent_layer.inputs_cache,
            candidate_inputs,
            parent_layer.weight
        )

        requests = []
        for input, target, prompt in zip(*args):
            input = parent_layer.instantiate_template([input], prompt=prompt)[0]
            requests.append(ScoreRequest(input, target, payload=target))

        prior_lps = self.scoring_lm.compute_log_p(
            requests,
            output_classes=parent_layer.output_classes,
        ).logp_targets.reshape(
            batch_size,
            1,
            num_samples,
        ).squeeze()

        if candidate_inputs_logps is None:
            candidate_inputs_logps = 0.

        input_scores = np.exp(lps + prior_lps - candidate_inputs_logps) \
            / np.exp(lps + prior_lps - candidate_inputs_logps).sum(1, keepdims=True)
        return input_scores


class AccuracyScorer(Scorer):
    def score_prompts(self, candidate_prompts, y, y_weights):
        from postprocessing import postprocess_prediction

        (num_samples,) = candidate_prompts.shape
        batch_size, num_targets = y.shape

        args = prepare_prompts_scoring_args(
            self.base_layer.inputs_cache, y, candidate_prompts
        )
        # build up a set of score requests
        requests = []
        for input, _, prompt in zip(*args):
            requests.append(
                self.base_layer.instantiate_template(input=input, prompt=prompt)
            )

        outputs = self.scoring_lm.generate(
            requests,
            stop=["\n\n"],
            temperature=0.0,
            max_tokens=10,
        )

        targets = np.array([postprocess_prediction(t) for t in targets])
        outputs = np.array([postprocess_prediction(yi) for yi in y])
        accuracy = np.array([t == y for t, y in zip(targets, outputs)])
        prompt_weights = accuracy.reshape(batch_size, num_targets, num_samples)
        prompt_scores = (y_weights[:, :, None] * prompt_weights).sum(1).mean(0)
        return prompt_scores

    def score_inputs(self, *args, **kwargs):
        raise NotImplementedError("Cannot score inputs with accuracy!")
