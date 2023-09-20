import logging
from typing import Iterable, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
from contextlib import contextmanager
from engine.template import DLNTemplate
from engine.network import NetworkNode
from engine.loss import LLoss


cache_forward_pass = True


def set_cache_enabled(enabled):
    global cache_forward_pass

    cache_forward_pass = enabled


@contextmanager
def cache_disable():
    global cache_forward_pass
    cache_forward_pass = False
    yield
    cache_forward_pass = True


class LanguageLayer(NetworkNode, ABC):
    def __init__(self):
        super().__init__()

        self._forward_lm = None
        self._backward_lm = None
        self._scoring_lm = None
        self._prompt_sampler = None
        self._hidden_sampler = None

    @abstractmethod
    def backward(
        self,
        y: np.array,
        y_weights: np.ndarray = None,
        losses: np.ndarray = None,
        num_p_samples: int = 1,
        num_h_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_forward_template(self):
        """Get the forward template for this layer."""
        pass

    @abstractmethod
    def instantiate_template(
        self, inputs: Iterable[str], **template_overrides
    ) -> List[str]:
        """Instantiate the template for this layer.

        Args:
            inputs: inputs to the template, will fill the field "input" in the template

        Returns:
            List[str]: instantiated templates
        """
        pass

    @property
    def forward_lm(self):
        if not self._forward_lm:
            from engine.ops import LanguageLayerOps

            self._forward_lm = LanguageLayerOps().forward_lm
        return self._forward_lm

    @property
    def backward_lm(self):
        if not self._backward_lm:
            from engine.ops import LanguageLayerOps

            self._backward_lm = LanguageLayerOps().backward_lm
        return self._backward_lm

    @property
    def scoring_lm(self):
        # forward lm is the default scoring lm
        if not self._scoring_lm:
            from engine.ops import LanguageLayerOps

            self._scoring_lm = LanguageLayerOps().scoring_lm
        return self._scoring_lm

    @abstractmethod
    def forward(self, inputs, **kwargs):
        pass

    @property
    def prompt_sampler(self):
        if not self._prompt_sampler:
            raise ValueError("Did you set a prompt sampler for this layer?")
        return self._prompt_sampler

    @property
    def hidden_sampler(self):
        return self._hidden_sampler

    @property
    def scorer(self):
        if not self._scorer:
            raise ValueError("Did you set a scorer for this layer?")
        return self._scorer

    def with_sampling_strategy(self, prompt_sampler, hidden_sampler=None):
        self._prompt_sampler = prompt_sampler
        self._hidden_sampler = hidden_sampler
        self._prompt_sampler.register_base_layer(self)
        if self._hidden_sampler:
            self._hidden_sampler.register_base_layer(self)
        return self

    def with_engine(self, engine_config):
        self.with_sampling_strategy(
            engine_config.prompt_sampler,
            engine_config.hidden_sampler
        )
        self.with_scoring_strategy(engine_config.scorer)
        return self

    def with_scoring_strategy(self, scorer):
        self._scorer = scorer
        scorer.register_base_layer(self)
        return self

    def with_forward_lm(self, forward_lm):
        self._forward_lm = forward_lm
        return self

    def with_backward_lm(self, backward_lm):
        self._backward_lm = backward_lm
        return self

    def with_score_lm(self, scoring_lm):
        self._scoring_lm = scoring_lm
        return self


class BaseLayer(LanguageLayer):
    def get_forward_template(self):
        if self.template_type == "prefix":
            return DLNTemplate(
                template="""{{ prompt }}

{{ input }}

{{ output_formatting_instruction }}
"""
            )
        elif self.template_type == "suffix":
            return DLNTemplate(
                template="""{{ input }}

{{ prompt }}

{{ output_formatting_instruction }}
"""
            )
        else:
            raise ValueError("Template type should be either suffix or prefix.")

    def __init__(
        self,
        template_type="prefix",
        init="",
        output_formatting_instruction="",
        output_classes=None,
    ):
        super().__init__()
        self.weight = init
        self.template_type = template_type
        self.forward_template = self.get_forward_template()
        self.output_formatting_instruction = output_formatting_instruction
        self.output_classes = output_classes
        self.requires_input = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def instantiate_template(self, inputs, **template_kwargs) -> List[str]:
        if "prompt" not in template_kwargs:
            template_kwargs["prompt"] = self.weight
        if "output_formatting_instruction" not in template_kwargs:
            template_kwargs[
                "output_formatting_instruction"
            ] = self.output_formatting_instruction
        return [
            self.forward_template.render(input=input, **template_kwargs)
            for input in inputs
        ]

    def forward(
        self,
        inputs=None,
        temperature=0.0,
    ) -> np.array:
        """Forward pass throught this layer.

        Args:
            output_classes: if not None, compute the constrained forward pass on the output classes, pick the highest probability amongst
                            the prototypes.
            temperature: temperature to use for the forward pass
            strip_double_newlines: if True, strip any "\n\n" that might have been added
            max_tokens: cap the max length for the forward pass
        """
        if inputs is None:
            inputs = self.input_nodes[0].outputs_cache

        tpl_inputs = self.instantiate_template(inputs)

        if self.output_classes is None:
            outputs = self.forward_lm.generate(
                tpl_inputs,
                stop=self.forward_template.stop_tokens,
                temperature=temperature,
            )
        else:
            if self.forward_lm.has_log_probs:
                from engine.scorer import ScoreRequest

                # dummy targets
                targets = [self.output_classes.prototype(0) for _ in inputs]
                # build up a set of score requests
                requests = []
                for input, target in zip(tpl_inputs, targets):
                    requests.append(ScoreRequest(input, target, payload=target))

                lps = self.forward_lm.compute_log_p(
                    requests, output_classes=self.output_classes, agg="sum"
                ).distribution
                # best output class index
                best_output_class_index = np.argmax(lps, axis=1)
                # get the best output class token
                outputs = [
                    self.output_classes.prototype(idx)
                    for idx in best_output_class_index
                ]
            else:
                logit_bias = {}
                max_len = 0

                for i in range(len(self.output_classes)):
                    token_ids = self.forward_lm.encode(self.output_classes.prototype(i))
                    max_len = max(max_len, len(token_ids))
                    assert max_len == 1

                    logit_bias[token_ids[0]] = 100

                outputs = self.forward_lm.generate(
                    tpl_inputs,
                    stop=self.forward_template.stop_tokens,
                    temperature=temperature,
                    max_tokens=max_len,
                    logit_bias=logit_bias,
                )

        if cache_forward_pass:
            self.inputs_cache = np.asarray(inputs)
            self.outputs_cache = np.asarray(outputs)
        return np.array(outputs)


class ResidualLayer(BaseLayer):
    residual_template = DLNTemplate(
        template="""{{ residual }}
Your thoughts were:
{{ input }}
"""
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info(
            "Residual template:\n" + f"{repr(self.residual_template.template)}"
        )

    def forward(self, inputs=None) -> np.array:
        if len(self.input_nodes) > 1:
            raise ValueError("ResidualLayer only implemented for 1 input node.")

        if inputs is None:
            inputs = self.input_nodes[0].outputs_cache

        if len(self.input_nodes):
            residual = self.input_nodes[0].inputs_cache
            inputs_plus_residual = self._apply_residual(inputs, residual)
        else:
            residual = None
            inputs_plus_residual = inputs

        outputs = super().forward(inputs_plus_residual)

        if cache_forward_pass:
            self.inputs_cache_original = inputs
            self.inputs_cache = inputs_plus_residual
            self.residual_cache = residual
            self.outputs_cache = outputs
        return outputs

    def backward(
        self,
        y: np.ndarray,
        y_weights: np.ndarray = None,
        losses: np.ndarray = None,
        targets: np.array = None,
        num_p_samples: int = 1,
        num_h_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a matrix of size (batch_size, num_outputs), and their corresponding scores.

        Args:
            y_weights: (batch_size, num_outputs)
        """
        if y is None:
            y_best = self.outputs_cache
        else:
            if type(y) == list:
                y = np.asarray(y)
            if y.ndim == 1:
                y = y[:, None]
            if y_weights is None:
                y_weights = np.ones((y.shape[0], 1))
            if y_weights.ndim == 1:
                y_weights = y_weights[:, None]

            y_best_ind = np.argmax(y_weights, axis=1)
            y_best = np.asarray([y[i, y_ind] for i, y_ind in enumerate(y_best_ind)])

        candidate_prompts = self.prompt_sampler.rewrite_prompts(
            y_best,
            losses=losses,
            num_samples=num_p_samples,
        )
        candidate_prompts_scores = self.scorer.score_prompts(
            candidate_prompts,
            y,
            y_weights,
            targets=targets,
            losses=losses,
        )
        best_prompt = candidate_prompts[candidate_prompts_scores.argmax()]

        logging.info("Candidate prompts:")
        for prompt, weight in zip(candidate_prompts, candidate_prompts_scores):
            logging.info("Prompt ({:.2f}) --> {}".format(weight, prompt))

        logging.info(
            "Best Prompt [{}], ({:.2f}) --> {}".format(
                candidate_prompts_scores.argmax(),
                candidate_prompts_scores.max(),
                best_prompt,
            )
        )

        if self.requires_input and self.hidden_sampler is not None:
            candidate_inputs_struct = self.hidden_sampler.rewrite_inputs(
                y_best, inputs=self.inputs_cache_original, num_samples=num_h_samples
            )
            candidate_inputs = candidate_inputs_struct.inputs

            # Now evaluate the probability of each sample to give the correct answer
            # the evaluation must be done with the residual layer
            candidate_inputs_residual = []
            for k in range(candidate_inputs.shape[1]):
                candidate_inputs_residual.append(
                    self._apply_residual(candidate_inputs[:, k], self.residual_cache)
                )
            candidate_inputs_residual = np.stack(candidate_inputs_residual, axis=1)
            candidate_inputs_scores = self.scorer.score_inputs(
                candidate_inputs_residual, y_best, candidate_inputs_struct.inputs_logps
            )
        else:
            candidate_inputs, candidate_inputs_scores = None, None

        self.weight = best_prompt
        self.candidate_prompts = candidate_prompts
        self.candidate_prompts_scores = candidate_prompts_scores
        return candidate_inputs, candidate_inputs_scores

    def _apply_residual(
        self,
        inputs: np.array,
        residual: np.array,
    ) -> np.array:
        """Apply a residual layer to the inputs."""
        if residual is None:
            return inputs

        residual_inputs = []
        for input, residual in zip(inputs, residual):
            residual_inputs.append(
                self.residual_template.render(
                    input=input,
                    residual=residual
                )
            )
        return np.asarray(residual_inputs)
