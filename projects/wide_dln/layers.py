from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Set
import numpy as np

from dln.operator import LLM
from dln.template import load_template
from dln.vi.utils import log_message


class Value:
    def __init__(self, data, _children: Optional[Set] = None, _op=""):
        self.data = data
        self.grad = ""
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        if _children is None:
            _children = set()
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for debugging

    # Define some basic str methods and operators on top of the Value.data
    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.data == other
        if isinstance(other, Value):
            return self.data == other.data
        return NotImplemented

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Value({self.data}, {self.grad}, {self._op})"

    # def replace(self, old, new):
    #     return Value(self.data.replace(old, new))

    # def __add__(self, other):
    #     if isinstance(other, str):
    #         other = Value(other, _op="")
    #     if isinstance(other, Value):
    #         return Value(self.data + other, (self, other), _op="+")
    #     return NotImplemented


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Node(Module):

    def __init__(self, init, forward_template, forward_evaluate): # TODO: better defaults
        self.prompt = Value(init, _op="prompt")
        self.forward_template = forward_template
        self.forward_evaluate = forward_evaluate

    def _build_prev(self, x):
        prev = {self.prompt}
        if isinstance(x, str):
            x = [x]
        for i in x:
            if isinstance(i, str):
                prev.add(Value(i, _op="input"))
            elif isinstance(i, Value):
                prev.add(i)
            else:
                raise ValueError(f"Invalid input type: {type(i)}")
        return prev

    def __call__(self, x, **kwargs):
        tpl_inputs = [
            self.forward_template.render(input=i, prompt=self.prompt)
            for i in x
        ]
        fwd_outputs = self.forward_evaluate(
            tpl_inputs,
            stop=self.forward_template.stop_tokens,
            **kwargs,
        )
        outputs = [
            Value(o, self._build_prev(i), _op="output")
            for i, o in zip(x, fwd_outputs)
        ]
        return np.asarray(outputs)

    @property
    def parameters(self):
        return [self.prompt]

    def __repr__(self):
        return f"Node({repr(self.prompt)})"


class BaseLayer(ABC):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        init: Optional[List[str]] = None,
        # num_nodes: Optional[int] = None,
        **kwargs,
    ):
        # TODO: raise an error if both init and num_nodes are None or if the init have different length than num_nodes
        forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.nodes: Node = [
            Node(i, forward_template, forward_evaluate) for i in init
        ]
        self.forward_evaluate = forward_evaluate # TODO: do we need this?

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        outputs = [node(inputs, **kwargs) for node in self.nodes]
        return np.asarray(outputs)

    @property
    def parameters(self):
        return [node.parameters for node in self.nodes]

    def __repr__(self):
        return f"Layer({repr(self.nodes)})"


class AggregationLayer(BaseLayer):

    def __init__(self, forward_evaluate, forward_template, init="", **kwargs):
        super().__init__(forward_evaluate, forward_template, init=[init], **kwargs)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        return super().forward(inputs.T, **kwargs).reshape(-1)
        # concat_inputs = ['\n'.join([o for o in row]) for row in inputs.T]
        # return super().forward(concat_inputs, **kwargs)


class WideLayer(BaseLayer):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        width: int = 1,
        init: list = None,
        **kwargs,
    ):
        assert width >= 1
        init = [""] * width if init is None else init
        assert len(init) == width
        super().__init__(forward_evaluate, forward_template, init=init, **kwargs)

# class LanguageNetwork(ABC):

#     def __init__(self, forward_evaluate, backward_evaluate):
#         self.forward_evaluate = forward_evaluate
#         self.backward_evaluate = backward_evaluate

#     @abstractmethod
#     def forward(self, x):
#         pass

#     def backward(self, loss):
#         pass


# class DeepWide(LanguageNetwork):
class DeepWide():
    def __init__(self, forward_evaluate, backward_evaluate):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.wide = WideLayer(forward_evaluate, "wide_forward", 2)
        self.agg = AggregationLayer(
            forward_evaluate,
            "agg_forward",
            init="Therefore, the answer is:",
        )
        self.h = None
        self.out = None

    def forward(self, x):
        h = self.wide(x)
        out = self.agg(h)
        return out

    def backward(self, loss):
        pass
        # p1s = sample_p1s()
        # pi_1: prompt agg
        # pi_0: prompt wide
        # h1: hidden state
        #
        # pi_1 proposal:
        # to update pi_1, we neet to sample agg prompt using
        # pi_1s = LLM(Template_pi_1(loss, h1))
        # pi_1_tild = argmax(score(llm(h1, c), y))
        # Basically, we obtain a set of C of pi_1 proposals,
        # and we select the one that maximizes the probability of the target.
        #
        # h1 proposal:
        # h1s = LLM(Template_h1(loss, pi_1))
        # h1_tild = argmax(score(llm(c, pi_1_tild), y))
        # Basically, we obtain a set of C of h1 proposals,
        # and we select the one that maximizes the probability of h1_tild.
        #
        # pi_0 proposal:
        # pi_0s = LLM(Template_pi_0(h1, h1_tild, x))
        # pi_0_tild = argmax(score(llm(x, c), h1_tild))

