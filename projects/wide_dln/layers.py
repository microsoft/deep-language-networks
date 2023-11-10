from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np

from dln.operator import LLM
from dln.template import load_template
from dln.vi.utils import log_message


class BaseLayer(ABC):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        init: Optional[str] = "",
        **kwargs,
    ):
        self.weight = init
        self.forward_template = load_template(forward_template, template_directory="./templates")
        log_message("Forward template:\n", f"{repr(self.forward_template.template)}")
        self.forward_evaluate = forward_evaluate

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        pass


class OutputLayer(BaseLayer):

    def forward(self, *args, **kwargs):
        pass


class AggregationLayer(BaseLayer):

    def __init__(self, forward_evaluate, forward_template, init="", **kwargs):
        super().__init__(forward_evaluate, forward_template, init=init, **kwargs)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        concat_inputs = ['\n'.join(row) for row in inputs.T]
        tpl_inputs = [
            self.forward_template.render(input=input, prompt=self.weight)
            for input in concat_inputs
        ]
        # log_message(tpl_inputs[0])
        outputs = self.forward_evaluate(
            tpl_inputs,
            stop=self.forward_template.stop_tokens,
            **kwargs,
        )
        return np.asarray(outputs)


class WideLayer(BaseLayer):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        width: int = 1,
        init: list = None,
        **kwargs,
    ):
        super().__init__(forward_evaluate, forward_template, **kwargs)
        assert width >= 1
        if init is None:
            init = [""] * width
        assert len(init) == width
        self.width = width
        weight = []
        for i in range(self.width):
            weight.append(init[i])
        self.weight = weight

    def forward_branch(
        self,
        inputs,
        branch_id,
        temperature=0.0,
        strip_double_newlines=True,
        max_tokens=256,
    ) -> np.array:
        """Forward pass throught this layer.

        Args:
            temperature: temperature to use for the forward pass
            strip_double_newlines: if True, strip any "\n\n" that might have been added
            max_tokens: cap the max length for the forward pass
        """
        assert 0 <= branch_id < self.width
        tpl_inputs = [
            self.forward_template.render(input=input, prompt=self.weight[branch_id])
            for input in inputs
        ]
        # log_message(tpl_inputs[0])
        outputs = self.forward_evaluate(
            tpl_inputs,
            stop=self.forward_template.stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            async_generation=True,
        )
        # strip any "\n\n" that might have been added
        if strip_double_newlines:
            outputs = [o.replace("\n\n", "\n") for o in outputs]
        return np.asarray(outputs)

    def forward(
        self,
        inputs,
        temperature=0.0,
        strip_double_newlines=True,
        max_tokens=256,
    ) -> np.array:
        outputs = []
        for branch_id in range(self.width):
            outputs.append(self.forward_branch(
                inputs, branch_id,
                temperature=temperature,
                strip_double_newlines=strip_double_newlines,
                max_tokens=max_tokens
            ))
        return np.array(outputs)


class DeepWide:
    def __init__(self, forward_evaluate, backward_evaluate, loss_fn):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.loss_fn = loss_fn
        self.wide = WideLayer(forward_evaluate, "wide_forward", 2)
        self.agg = AggregationLayer(
            forward_evaluate,
            "agg_forward",
            init="Therefore, the answer is:",
        )

    def forward(self, x):
        x = self.wide(x)
        x = self.agg(x)
        return x

    def backward(self, loss):
        pass
