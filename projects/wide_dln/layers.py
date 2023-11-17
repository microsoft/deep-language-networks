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

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Value({self.data}, {self.grad}, {self._op})"

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()


class ModuleMixin(ABC):

    def zero_grad(self):
        for p in self.parameters():
            p.grad = ""

    @abstractmethod
    def parameters(self):
        pass


class Node(ModuleMixin):

    def __init__(self, init, forward_template, forward_evaluate):
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

    def parameters(self):
        return [self.prompt]

    def __repr__(self):
        return f"Node({repr(self.prompt)})"


class BaseLayer(ModuleMixin, ABC):

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

    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]

    def __repr__(self):
        return f"Layer({repr(self.nodes)})"


class AggregationLayer(BaseLayer):

    def __init__(self, forward_evaluate, forward_template, init="", **kwargs):
        super().__init__(forward_evaluate, forward_template, init=[init], **kwargs)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        return super().forward(inputs.T, **kwargs).reshape(-1)


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


class DeepWideNetwork(ModuleMixin):
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
        self.h = self.wide(x)
        self.out = self.agg(self.h)
        return self.out

    def backward(self, loss):
        self.zero_grad()
        for output in self.out:
            output.backward()
        pass

    def parameters(self):
        return self.wide.parameters() + self.agg.parameters()


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
