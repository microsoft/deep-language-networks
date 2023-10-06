from dataclasses import dataclass
from typing import List
import numpy as np

from abc import ABC, abstractmethod
from engine.ops import GPT, ScoreRequest
from engine.network import NetworkNode
from engine.layers import LanguageLayer
import logging


def prepare_prompts_scoring_args(
    base_layer: LanguageLayer,
    inputs: np.array,
    outputs: np.array,
    prompts: np.array,
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
    if inputs.ndim != 1:
        inputs = inputs.squeeze()
    if inputs.ndim != 1:
        raise ValueError("Expected inputs to be 1D.")

    num_prompts = prompts.shape[0]
    batch_size = inputs.shape[0]
    num_labels = outputs.shape[1]

    # now instantiate the forward template for each of the prompt candidates
    # this proceeds by instantiating one candidate for each element in the batch
    instantiated_inputs = np.empty((batch_size, num_prompts), dtype=object)
    for i in range(num_prompts):
        instantiated_inputs[:, i] = base_layer.instantiate_template(
            inputs, prompt=prompts[i]
        )

    evals = []
    for i in range(batch_size):
        for j in range(num_labels):
            for k in range(num_prompts):
                evals.append(
                    ScoreRequest(
                        context=instantiated_inputs[i, k],
                        target=outputs[i, j],
                        payload=outputs[i, j],
                    )
                )
    return evals


def prepare_inputs_scoring_args(
    base_layer: LanguageLayer, inputs: np.array, outputs: np.array, prompt: str
):
    """
    Args:
        inputs: (batch_size, num_inputs)
        outputs: (batch_size, 1) or (batch_size, num_outputs)
        prompt: str

    Returns
        (batch_size, num_inputs, num_outputs)
    """

    # add a dimension in case there is only 1 output
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    if inputs.ndim == 1:
        inputs = inputs[:, None]

    batch_size = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_labels = outputs.shape[1]

    # now instantiate the forward template for each of the input candidates
    # this proceeds by instantiating one candidate for each element in the batch
    instantiated_inputs = np.empty((batch_size, num_inputs), dtype=object)
    for i in range(num_inputs):
        instantiated_inputs[:, i] = base_layer.instantiate_template(
            inputs[:, i], prompt=prompt
        )

    evals = []
    for i in range(batch_size):
        for j in range(num_inputs):
            for k in range(num_labels):
                evals.append(
                    ScoreRequest(
                        context=instantiated_inputs[i, j],
                        target=outputs[i, k],
                        payload=outputs[i, k],
                    )
                )
    return evals


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
    def score_prompts(self, candidate_prompts, y, weights, targets=None):
        pass

    @abstractmethod
    def score_inputs(self, candidate_inputs, y, candidate_inputs_logps=None):
        pass


class LogProbsScorer(Scorer):
    def __init__(self, logp_penalty=0.0):
        self.logp_penalty = logp_penalty

    def score_prompts(
        self,
        candidate_prompts: np.array,
        y: np.array,
        y_weights: np.ndarray,
        targets=None,
        losses=None,
    ):
        (num_samples,) = candidate_prompts.shape
        batch_size, num_targets = y.shape

        # build up a set of score requests
        requests = prepare_prompts_scoring_args(
            self.base_layer, self.base_layer.inputs_cache, y, candidate_prompts
        )
        lps = self.scoring_lm.compute_log_p(
            requests,
            output_classes=self.base_layer.output_classes,
        ).logp_targets.reshape(batch_size, num_targets, num_samples)

        logp_prompts = (y_weights[:, :, None] * lps).sum(1).mean(0)

        # compute the logp of the layer's own outputs for the logp_penalty
        is_output_layer = len(self.base_layer.output_nodes) == 0
        if self.logp_penalty > 0.0 and not is_output_layer:
            # build up a set of score requests
            requests = prepare_prompts_scoring_args(
                self.base_layer,
                self.base_layer.inputs_cache,
                self.base_layer.outputs_cache,
                candidate_prompts,
            )
            lps = self.scoring_lm.compute_log_p(
                requests,
                output_classes=self.base_layer.output_classes,
            ).logp_targets.reshape(batch_size, num_samples)

            # assume loss is 1 or 0
            error_indices = np.where(losses > 0)[0]
            if len(error_indices) == 0:
                lp_penalty = 0.0
            else:
                lp_penalty = lps[error_indices].sum(0) / len(error_indices)
            return logp_prompts - self.logp_penalty * lp_penalty
        else:
            return logp_prompts

    def score_inputs(self, candidate_inputs, y, candidate_inputs_logps=None):
        batch_size, num_samples = candidate_inputs.shape

        requests = prepare_inputs_scoring_args(
            self.base_layer, candidate_inputs, y, self.base_layer.weight
        )

        lps = self.scoring_lm.compute_log_p(
            requests,
            self.base_layer.output_classes,
        ).logp_targets.reshape(
            batch_size,
            num_samples,
        )

        input_scores = np.exp(lps) / np.exp(lps).sum(1, keepdims=True)
        return input_scores


class VIScorer(LogProbsScorer):
    def score_inputs(self, candidate_inputs, y, candidate_inputs_logps=None):
        batch_size, num_samples = candidate_inputs.shape
        if y.ndim == 1:
            y = y[:, None]

        num_output_targets = y.shape[1]
        assert num_output_targets == 1

        requests = prepare_inputs_scoring_args(
            self.base_layer, candidate_inputs, y, self.base_layer.weight
        )
        lps = (
            self.scoring_lm.compute_log_p(
                requests,
                self.base_layer.output_classes,
                agg="sum",
            )
            .logp_targets.reshape(
                batch_size,
                num_samples,
                num_output_targets,
            )
            .squeeze()
        )

        parent_layer = self.base_layer.input_nodes[0]
        requests = prepare_inputs_scoring_args(
            parent_layer,
            parent_layer.inputs_cache,
            candidate_inputs,
            parent_layer.weight,
        )
        prior_lps = (
            self.scoring_lm.compute_log_p(
                requests,
                output_classes=parent_layer.output_classes,
            )
            .logp_targets.reshape(
                batch_size,
                1,
                num_samples,
            )
            .squeeze()
        )

        if candidate_inputs_logps is None:
            candidate_inputs_logps = 0.0

        input_scores = np.exp(lps + prior_lps - candidate_inputs_logps) / np.exp(
            lps + prior_lps - candidate_inputs_logps
        ).sum(1, keepdims=True)
        return input_scores


class AccuracyScorer(Scorer):
    def score_prompts(self, candidate_prompts, y, y_weights, targets=None, losses=None):
        from postprocessing import postprocess_prediction

        (num_samples,) = candidate_prompts.shape
        batch_size, num_targets = y.shape

        requests = prepare_prompts_scoring_args(
            self.base_layer, self.base_layer.inputs_cache, y, candidate_prompts
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


class FullStackScorer(Scorer):
    def score_prompts(self, candidate_prompts, y, y_weights, targets=None, losses=None):
        from engine.layers import cache_disable
        from postprocessing import postprocess_prediction

        with cache_disable():
            old_prompt = self.base_layer.weight
            prompt_accs = []
            for prompt in candidate_prompts:
                self.base_layer.weight = prompt

                outputs = self.base_layer.forward_graph(self.base_layer.inputs_cache)
                outputs = [postprocess_prediction(o) for o in outputs]
                targets = [postprocess_prediction(t) for t in targets]
                prompt_acc = np.array([o == t for o, t in zip(outputs, targets)]).mean()
                prompt_accs.append(prompt_acc)
            self.base_layer.weight = old_prompt
        return np.asarray(prompt_accs)

    def score_inputs(self, *args, **kwargs):
        return None
