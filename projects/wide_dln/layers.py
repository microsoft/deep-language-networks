from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Set
import numpy as np

from dln.operator import LLM
from dln.template import load_template


class Value:
    def __init__(
        self,
        data,
        prev_values: Optional[Set] = None,
        layer: "BaseLayer" = None,
        _op="",
    ):
        self.data = data
        self.grad = ""
        self.layer = layer
        self._op = _op # the op that produced this node, for debugging
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        if prev_values is None:
            prev_values = set()
        self.prev_values = set(prev_values)
        self.next_values = set()  # is set by next layer
        for v in self.prev_values:  # set next_values of prev_values
            v.next_values = v.next_values.union({self})

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Value({self.data}, {self.grad}, {self._op})"

    def _topological_order(self):
        # topological order all of the values in the graph
        # starting from the current value
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev_values:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self):
        topo = self._topological_order()
        for v in reversed(topo):
            v._backward()

    def prompt_layers_backward(self):
        """Returns all layers that have a prompt in the backward graph"""
        topo = self._topological_order()
        layers = []
        for v in reversed(topo):
            if v._op == "prompt" and v.layer is not None and v.layer not in layers:
                layers.append(v.layer)
        return layers

    def prev_layers(self):
        prev_layers = set()
        for v in self.prev_values:
            if v.layer is not None and v.layer != self.layer:
                prev_layers.append(v.layer)
        return set(prev_layers)

    def next_layers(self):
        next_layers = []
        for v in self.next_values:
            if v.layer is not None and v.layer != self.layer:
                next_layers.append(v.layer)
        return next_layers


class ModuleMixin(ABC):

    def zero_grad(self):
        for p in self.parameters():
            p.grad = ""
            p.layer.inputs = []
            p.layer.outputs = []

    @abstractmethod
    def parameters(self):
        pass


class Node(ModuleMixin):

    def __init__(self, init, forward_template, forward_evaluate, layer):
        self.prompt = Value(init, layer=layer, _op="prompt")
        self.forward_template = forward_template
        self.forward_evaluate = forward_evaluate
        self.layer = layer

    def _build_prev(self, x):
        prev = {self.prompt}
        if isinstance(x, str):
            x = [x]
        for i in x:
            if isinstance(i, str):
                prev.add(Value(i, layer=self.layer, _op="input"))
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
            Value(o, self._build_prev(i), layer=self.layer, _op="output")
            for i, o in zip(x, fwd_outputs)
        ]
        # build prompt and outputs backward

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
            Node(i, forward_template, forward_evaluate, self) for i in init
        ]
        self.forward_evaluate = forward_evaluate # TODO: do we need this?
        self.inputs = []
        self.outputs = []

        # self._forward_lm = None
        # self._backward_lm = None
        # self._scoring_lm = None
        # self._prompt_sampler = None
        # self._hidden_sampler = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        self.inputs = inputs
        outputs = [node(inputs, **kwargs) for node in self.nodes]
        self.outputs = outputs
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

    def backward(self, losses):
        print("AggregationLayer backward")


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

    def backward(self, losses):
        print("WideLayer backward")


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
        self.outputs = None

    def forward(self, x):
        self.h = self.wide(x)
        self.outputs = self.agg(self.h)
        return self.outputs

    def backward(self, losses):
        self.zero_grad()
        any_out = self.outputs[0]
        prompt_layers = any_out.prompt_layers_backward()
        for l in prompt_layers:
            l.backward(losses)

    def parameters(self):
        return self.wide.parameters() + self.agg.parameters()
