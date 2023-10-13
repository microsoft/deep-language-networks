from typing import List

import numpy as np

from dln.loss import ZeroOneLoss
from dln.operator import LLM
from dln.score import LogProbs, LogProbsScore, OutputClasses, ScoreRequest
from dln.template import load_template
from dln.vi.utils import log_message


class PriorLayer:

    def __init__(
        self,
        logprobs_score: LogProbsScore,
        forward_evaluate: LLM,
        forward_template: str,
        init: str = "",
    ):
        self.forward_template = load_template(
            forward_template
        )
        log_message("Forward template:\n", f"{repr(self.forward_template.template)}")
        self.weight = init
        self.logprobs_score = logprobs_score
        self.forward_evaluate = forward_evaluate

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        inputs,
        output_classes: OutputClasses = None,
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
        if output_classes is None:
            tpl_inputs = [
                self.forward_template.render(input=input, prompt=self.weight)
                for input in inputs
            ]
            log_message(tpl_inputs[0])
            outputs = self.forward_evaluate(
                tpl_inputs,
                stop=self.forward_template.stop_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
                async_generation=True,
            )
        else:
            if self.forward_evaluate.has_logprobs:
                # compute log p of each output class, second return value is the p(class)
                targets = [output_classes.prototype(0) for _ in inputs]
                lp = self.log_p(
                    inputs, targets, output_classes=output_classes, agg="sum"
                ).distribution
                # best output class index
                best_output_class_index = np.argmax(lp, axis=1)
                # get the best output class token
                outputs = [output_classes.prototype(idx) for idx in best_output_class_index]
            else:
                tpl_inputs = [
                    self.forward_template.render(input=input, prompt=self.weight)
                    for input in inputs
                ]
                logit_bias = {}
                max_len = 0

                for i in range(len(output_classes)):
                    token_ids = self.forward_evaluate.encode(output_classes.prototype(i))
                    max_len = max(max_len, len(token_ids))
                    assert max_len == 1
                    logit_bias[token_ids[0]] = 100

                outputs = self.forward_evaluate(
                    tpl_inputs,
                    stop=self.forward_template.stop_tokens,
                    temperature=temperature,
                    max_tokens=max_len,
                    logit_bias=logit_bias,
                )
        # strip any "\n\n" that might have been added
        if strip_double_newlines:
            outputs = [o.replace("\n\n", "\n") for o in outputs]
        return np.asarray(outputs)

    def log_p_request(self, input: str, target: str, prompt: str) -> ScoreRequest:
        # build up a set of score requests
        context = self.forward_template.render(input=input, prompt=prompt)
        return ScoreRequest(context=context, target=target, payload=target)

    def log_p(
        self,
        inputs: List[str],
        targets: List[str],
        prompts=None,
        output_classes=None,
        agg="max",
    ) -> LogProbs:
        requests = []

        if prompts is None:
            prompts = [self.weight for _ in inputs]

        for input, target, prompt in zip(inputs, targets, prompts):
            requests.append(self.log_p_request(input, target, prompt=prompt))

        # build up a set of score requests
        logprobs = self.logprobs_score.score_requests(requests, output_classes, agg=agg)
        return logprobs

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
                requests.append(self.forward_template.render(input=input, prompt=prompt))

        # build up a set of score requests
        outputs = self.forward_evaluate(
            requests,
            stop=self.forward_template.stop_tokens,
            temperature=1.0 if num_samples > 1 else 0.,
            max_tokens=max_tokens,
        )
        targets = np.array([t for t in targets] * num_samples)

        loss = ZeroOneLoss(postprocess_prediction)
        losses = loss(outputs, targets).reshape(-1, num_samples)
        accuracy = (1. - losses).mean(1)
        return accuracy


class ResidualPriorLayer(PriorLayer):

    def __init__(
        self,
        logprobs_score: LogProbsScore,
        forward_evaluate: LLM,
        forward_template,
        init="",
        residual_template="classify_residual"
    ):
        super().__init__(logprobs_score, forward_evaluate, forward_template, init=init)
        self.residual_template = load_template(residual_template)
        log_message("Residual template:\n", f"{repr(self.residual_template.template)}")

    def forward(self, inputs, **kwargs) -> np.array:
        outputs = super().forward(inputs, **kwargs)
        return outputs

    def apply_residual(
        self, outputs: np.array, inputs: np.array, use_template=False
    ) -> np.array:
        outputs_ = []
        if use_template:
            for output, input in zip(outputs, inputs):
                tpl_input = self.forward_template.render(
                    input=input, prompt=self.weight
                )
                outputs_.append(
                    self.residual_template.render(
                        input=tpl_input, output=output
                    )
                )
        else:
            for output, input in zip(outputs, inputs):
                outputs_.append(
                    self.residual_template.render(
                        input=input, output=output
                    )
                )
        return np.array(outputs_)
