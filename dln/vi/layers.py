from typing import List

import numpy as np

from dln.operator import forward_evaluate
from dln.score import LogProbs, LogProbsScore, OutputClasses, ScoreRequest
from dln.template import load_template
from dln.vi.utils import log_message


class PriorLayer:
    def __init__(self, forward_template, init=None):
        self.forward_template = load_template(
            forward_template
        )
        log_message("Forward template:\n", f"{repr(self.forward_template.template)}")
        self.weight = init

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
            outputs = forward_evaluate(
                tpl_inputs,
                stop=self.forward_template.stop_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
                async_generation=True,
            )
        else:
            # compute constrained forward pass on the output classes
            targets = [output_classes.prototype(0) for _ in inputs]
            # compute log p of each output class, second return value is the p(class)
            lp = self.log_p(
                inputs, targets, output_classes=output_classes, agg="sum"
            ).contexts
            # best output class index
            best_output_class_index = np.argmax(lp, axis=1)
            # get the best output class token
            outputs = [output_classes.prototype(idx) for idx in best_output_class_index]
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
        logprobs = LogProbsScore().score_requests(requests, output_classes, agg=agg)
        return logprobs


class ResidualPriorLayer(PriorLayer):

    def __init__(self, forward_template, init=None, residual_template="classify_residual"):
        super().__init__(forward_template, init=init)
        self.residual_template = load_template(
            residual_template
        )
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
