from typing import List, Tuple

import numpy as np
from dln.loss import ZeroOneLoss

import dln.operator
from dln.operator import forward_evaluate
from dln.vi.sampler import PosteriorSampler, PromptSampler
from dln.score import LogProbs, LogProbsScore, OutputClasses, ScoreRequest
from dln.template import load_template, DLNTemplate

from dataclasses import dataclass
import logging


@dataclass
class ForwardInfos:
    layers: np.ndarray


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


class BaseLayer:
    @classmethod
    def get_backwards_inputs_template(cls):
        return DLNTemplate(
            template="""A student is completing a task that requires producing a text output from a text input.
In addition to the input, the student receives an instruction that explains the task.
Your job is to transform the input to help the student generate the correct output.

## Instruction:
> {{ prompt }}

## Input:
> {{ input }}

## Correct Output:
> {{ y }}

Write an improved version of the input so that that the student will generate the correct output when following the instruction.

## Improved Input:
>"""
    )

    @classmethod
    def get_forward_template(cls):
        return DLNTemplate(
            template="""{{ prompt }}

{{ input }}

{{ output_formating_instruction }}
"""
        )

    def __init__(
        self,
        init="",
        output_formating_instruction="",
        output_classes=None,
        score_method="logprobs",
    ):
        self.weight = init
        self.forward_template = self.get_forward_template()
        self.output_formating_instruction = output_formating_instruction
        self.output_classes = output_classes
        self.hidden_sampler = PosteriorSampler(
            self.get_backwards_inputs_template(),
        )
        self.prompt_sampler = PromptSampler()
        self.score_method = score_method
        self.requires_input = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        inputs,
        temperature=0.0,
        strip_double_newlines=True,
        max_tokens=256,
    ) -> np.array:
        """Forward pass throught this layer.

        Args:
            output_classes: if not None, compute the constrained forward pass on the output classes, pick the highest probability amongst
                            the prototypes.
            temperature: temperature to use for the forward pass
            strip_double_newlines: if True, strip any "\n\n" that might have been added
            max_tokens: cap the max length for the forward pass
        """
        tpl_inputs = [
            self.forward_template.render(
                input=input,
                prompt=self.weight,
                output_formating_instruction=self.output_formating_instruction,
            )
            for input in inputs
        ]

        if self.output_classes is None:
            outputs = forward_evaluate(
                tpl_inputs,
                stop=self.forward_template.stop_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            if dln.operator.forward_interpreter.has_log_probs:
                # compute log p of each output class, second return value is the p(class)
                targets = [self.output_classes.prototype(0) for _ in inputs]
                lp = self.log_p(
                    inputs, targets, output_classes=self.output_classes, agg="sum"
                ).distribution
                # best output class index
                best_output_class_index = np.argmax(lp, axis=1)
                # get the best output class token
                outputs = [
                    self.output_classes.prototype(idx)
                    for idx in best_output_class_index
                ]
            else:
                logit_bias = {}
                max_len = 0

                for i in range(len(self.output_classes)):
                    token_ids = dln.operator.forward_interpreter.encode(
                        self.output_classes.prototype(i)
                    )
                    max_len = max(max_len, len(token_ids))
                    assert max_len == 1
                    logit_bias[token_ids[0]] = 100

                outputs = forward_evaluate(
                    tpl_inputs,
                    stop=self.forward_template.stop_tokens,
                    temperature=temperature,
                    max_tokens=max_len,
                    logit_bias=logit_bias,
                )
        # strip any "\n\n" that might have been added
        if strip_double_newlines:
            outputs = [o.replace("\n\n", "\n") for o in outputs]

        self.outputs_cache = np.asarray(outputs)
        self.inputs_cache = np.asarray(inputs)

        return self.outputs_cache

    def log_p_request(self, input: str, target: str, prompt: str) -> ScoreRequest:
        # build up a set of score requests
        context = self.forward_template.render(
            input=input,
            prompt=prompt,
            output_formating_instruction=self.output_formating_instruction,
        )
        return ScoreRequest(context=context, target=target, payload=target)

    def log_p(
        self,
        inputs: List[str],
        targets: List[str],
        prompts=None,
        output_classes=None,
        agg="sum",
    ) -> LogProbs:
        requests = []

        if prompts is None:
            prompts = [self.weight for _ in inputs]

        for input, target, prompt in zip(inputs, targets, prompts):
            requests.append(self.log_p_request(input, target, prompt=prompt))

        # build up a set of score requests
        logprobs = LogProbsScore().score_requests(requests, output_classes, agg=agg)
        return logprobs

    def _get_best_prompts(
        self,
        y,
        y_weights: np.ndarray = None,
        losses: np.ndarray = None,
        num_samples: int = 1,
    ):
        y_max_ind = np.argmax(y_weights, axis=1)
        y_max = np.asarray([y[i, y_ind] for i, y_ind in enumerate(y_max_ind)])
    
        candidate_prompts: np.array = self.prompt_sampler.sample_q_p(
            inputs=self.inputs_cache,
            y=y_max,
            y_hat=self.outputs_cache,
            losses=losses,
            prompt=self.weight,
            num_samples=num_samples,
            held_out_half=False,
        )
        args = prepare_prompts_scoring_args(self.inputs_cache, y, candidate_prompts)

        if self.score_method == "logprobs":
            prompt_weights = self.log_p(
                *args, output_classes=self.output_classes, agg="sum"
            ).logp_targets
        elif self.score_method == "accuracy":
            prompt_weights = self.accuracy(
                *args,
                num_samples=1,
            )
        prompt_weights = prompt_weights.reshape(
            self.inputs_cache.shape[0], y.shape[1], num_samples
        )
        prompt_scores = (y_weights[:, :, None] * prompt_weights).sum(1).mean(0)
        best_prompt = candidate_prompts[np.argmax(prompt_scores)]
        return best_prompt, candidate_prompts, prompt_scores

    def _get_best_inputs(
        self,
        y,
        y_weights: np.ndarray = None,
        prompt: str = None,
        losses: np.ndarray = None,
        num_samples: int = 1,
    ):
        """ """
        if prompt is None:
            prompt = self.weight

        y_max_ind = np.argmax(y_weights, axis=1)
        y_max = np.asarray([y[i, y_ind] for i, y_ind in enumerate(y_max_ind)])

        # now re-write the inputs
        new_inputs, _ = self.hidden_sampler.sample_q_h(
            x=self.inputs_cache,
            y=y_max,
            h=y_max,
            prompt=prompt,
            next_prompt=prompt,
            num_samples=num_samples,
            return_logprobs=True,
        )

        # Now evaluate the probability of each sample to give the correct answer
        args = prepare_inputs_scoring_args(new_inputs, y_max, prompt)
        if self.score_method == "logprobs":
            inputs_weights = self.log_p(
                *args, output_classes=self.output_classes, agg="sum"
            ).logp_targets
        elif self.score_method == "accuracy":
            inputs_weights = self.accuracy(
                *args,
                num_samples=1,
            )
        inputs_weights = inputs_weights.reshape(
            self.inputs_cache.shape[0],
            new_inputs.shape[1],
        )
        inputs_weights = np.exp(inputs_weights) / np.exp(inputs_weights).sum(1, keepdims=True)
        return new_inputs, inputs_weights

    def backward(
        self,
        y: np.ndarray,
        y_weights: np.ndarray = None,
        losses: np.ndarray = None,
        num_p_samples: int = 1,
        num_h_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a matrix of size (batch_size, num_outputs), and their corresponding scores.

        Args:
            y_weights: (batch_size, num_outputs)
        """
        if type(y) == list:
            y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        if y_weights is None:
            y_weights = np.ones((y.shape[0], 1))
        if y_weights.ndim == 1:
            y_weights = y_weights[:, None]

        new_prompt, prompts, weights = self._get_best_prompts(y, y_weights, losses, num_p_samples)

        logging.info("Candidate prompts:")
        for prompt, weight in zip(prompts, weights):
            logging.info("Prompt ({:.2f}) --> {}".format(weight, prompt))

        logging.info("Best Prompt ({:.2f}) --> {}".format(weights.max(), new_prompt))
        logging.info("")

        if self.requires_input:
            new_inputs, new_weights = self._get_best_inputs(
                y, y_weights, new_prompt, losses, num_h_samples
            )
        else:
            new_inputs, new_weights = None, None

        self.weight = new_prompt
        self.prompts = prompts
        self.weights = weights

        return new_inputs, new_weights

    def accuracy(
        self,
        inputs: List[str],
        targets: List[str],
        prompts=None,
        num_samples=1,
        max_tokens=10,
        postprocess_prediction=None,
    ) -> LogProbs:
        requests = []

        if prompts is None:
            prompts = [self.weight for _ in inputs]

        for _ in range(num_samples):
            for input, _, prompt in zip(inputs, targets, prompts):
                requests.append(
                    self.forward_template.render(input=input, prompt=prompt)
                )

        # build up a set of score requests
        outputs = forward_evaluate(
            requests,
            stop=self.forward_template.stop_tokens,
            temperature=1.0 if num_samples > 1 else 0.0,
            max_tokens=max_tokens,
        )
        targets = np.array([t for t in targets] * num_samples)

        loss = ZeroOneLoss(postprocess_prediction)
        losses = loss(outputs, targets).reshape(-1, num_samples)
        accuracy = (1.0 - losses).mean(1)
        return accuracy


class ForwardLayer(BaseLayer):
    pass


class ResidualLayer(BaseLayer):
    residual_template = DLNTemplate(
        template="""{{ residual }}
Your thoughts were:
{{ input }}
"""
    )

    def __init__(
        self,
        init="",
        output_formating_instruction="",
        output_classes=None,
    ):
        super().__init__(
            init=init,
            output_formating_instruction=output_formating_instruction,
            output_classes=output_classes,
        )
        logging.info("Residual template:\n" + f"{repr(self.residual_template.template)}")

    def forward(self, inputs, residual=None) -> np.array:
        outputs = self._apply_residual(inputs, residual)
        outputs = super().forward(outputs)

        self.inputs_cache = inputs
        self.residual_cache = residual
        self.outputs_cache = outputs

        return outputs

    def _get_best_inputs(
        self,
        y,
        y_weights: np.ndarray = None,
        prompt: str = None,
        losses: np.ndarray = None,
        num_samples: int = 1,
    ):
        if prompt is None:
            prompt = self.weight

        y_max_ind = np.argmax(y_weights, axis=1)
        y_max = np.asarray([y[i, y_ind] for i, y_ind in enumerate(y_max_ind)])

        # now re-write the inputs
        new_inputs, _ = self.hidden_sampler.sample_q_h(
            x=self.inputs_cache,
            y=y_max,
            h=y_max,
            prompt=prompt,
            next_prompt=prompt,
            num_samples=num_samples,
            return_logprobs=True,
        )

        # Now evaluate the probability of each sample to give the correct answer
        # the evaluation must be done with the residual layer
        new_inputs_residual = []
        for k in range(new_inputs.shape[1]):
            new_inputs_residual.append(
                self._apply_residual(
                    new_inputs[:, k], self.residual_cache
                )
            )

        new_inputs_residual = np.stack(new_inputs_residual, axis=1)

        args = prepare_inputs_scoring_args(new_inputs_residual, y_max, prompt)
        if self.score_method == "logprobs":
            inputs_weights = self.log_p(
                *args, output_classes=self.output_classes, agg="sum"
            ).logp_targets
        elif self.score_method == "accuracy":
            inputs_weights = self.accuracy(
                *args,
                num_samples=1,
            )
        inputs_weights = inputs_weights.reshape(
            self.inputs_cache.shape[0], new_inputs.shape[1]
        )
        inputs_weights = np.exp(inputs_weights) / np.exp(inputs_weights).sum(1, keepdims=True)

        for i, ni, wi in zip(self.inputs_cache[:3], new_inputs[:3], inputs_weights[:3]):
            logging.info("Original sentence: " + i)
            for nii, wii in zip(ni, wi):
                logging.info("Rewrite ({:.2f}) --> {}".format(wii, nii))
            logging.info("")
        
        return new_inputs, inputs_weights

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
                self.residual_template.render(input=input, residual=residual)
            )
        return np.asarray(residual_inputs)


class StepLayer(ResidualLayer):
    @classmethod
    def get_forward_template(cls, step_number=None):
        return DLNTemplate(
            template=f"""{{ input }}

Step """ + str(step_number) + """. {{ prompt }}

{{ output_formating_instruction }}
""",
        )

    def __init__(
        self,
        init="",
        output_formating_instruction="",
        output_classes=None,
        step_number=1
    ):
        super().__init__(
            init=init,
            output_formating_instruction=output_formating_instruction,
            output_classes=output_classes,
        )
        self.forward_template = self.get_forward_template(step_number)
