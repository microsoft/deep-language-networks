from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import numpy as np

from dln.operator import LLM
from dln.template import load_template


@dataclass
class Info:
    input: str = None
    output: str = None
    target: str = None
    loss: float = 0.0


def loss_to_info(inputs, outputs, targets, loss):
    # TODO: move to loss.py
    return [
        Info(i, o, t, l)
        for i, o, t, l in zip(inputs, outputs, targets, loss)
    ]


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
        self.set_prev_layers()
        self.next_values = set()  # is set by next layer
        # set next_values of prev_values
        for v in self.prev_values:
            v.next_values = v.next_values.union({self})
            self.set_next_layers()

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

    def set_prev_layers(self):
        for v in self.prev_values:
            if (
                v.layer is not None
                and v.layer != self.layer
                and v.layer not in self.layer.prev_layers
            ):
                self.layer.prev_layers.append(v.layer)

    def set_next_layers(self):
        for v in self.prev_values:
            if (
                self.layer is not None
                and self.layer != v.layer
                and self.layer not in v.layer.next_layers
            ):
                v.layer.next_layers.append(self.layer)


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
        prompt_sampler: "Sampler",
        hidden_sampler: "Sampler",
        init: Optional[List[str]] = None,
        # num_nodes: Optional[int] = None,
        **kwargs,
    ):
        # TODO: raise an error if both init and num_nodes are None or
        # if the init have different length than num_nodes
        forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.prev_layers = []
        self.next_layers = []
        self.inputs = []
        self.outputs = []
        self.nodes: Node = [
            Node(i, forward_template, forward_evaluate, self) for i in init
        ]
        self.forward_evaluate = forward_evaluate # TODO: do we need this?
        self.prompt_sampler = prompt_sampler
        self.hidden_sampler = hidden_sampler

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

    def backward(self, losses):
        prompts = [n.prompt for n in self.nodes]
        new_prompts = self.prompt_sampler(prompts, losses)
        return new_prompts

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
        super().backward(losses)
        print("AggregationLayer backward")
        for i, o, l in zip(self.inputs, self.outputs, losses):
            print(f"input: {i}\noutput: {o}\nloss: {l}\n")
        print("AggregationLayer backward")
        # n_samples = 4
        # local_losses_info = []
        # for _input, loss in zip(self.inputs, losses):
        #     concat_input = "\n".join((map(str, _input)))
        #     local_losses_info.append(Info(
        #         str(concat_input),
        #         str(loss.output),
        #         str(loss.target),
        #         loss.loss,
        #     ))

        # # propose new prompts using agg_backward
        # bwd_template = load_template(
        #     "agg_backward",
        #     template_directory="./templates"
        # )
        # tpl_inputs = [
        #     bwd_template.render(prompt=self.nodes[0].prompt, backward_infos=losses)
        #     for _ in range(n_samples)
        # ]

        # bwd_outputs = backward_evaluate(
        #     tpl_inputs,
        #     stop=bwd_template.stop_tokens,
        #     # **kwargs,
        # )

        # select the one that maximizes the probability of the target
        # use the selected prompt to update all layer.nodes.prompt


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
        super().backward(losses)
        print("WideLayer backward")
        for i, o, l in zip(self.inputs, self.outputs, losses):
            print(f"input: {i}\noutput: {o}\nloss: {l}\n")
        print("WideLayer backward")


class LanguageNetwork(ModuleMixin, ABC):

    def backward_layers(self):
        if self.outputs is None:
            raise ValueError("No outputs to backpropagate")
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v.prev_layers:
                    build_topo(prev)
                topo.append(v)
        build_topo(self.outputs[0].layer)
        return reversed(topo)

    def backward(self, losses):
        for l in self.backward_layers():
            l.backward(losses)  # Do something with new prompts here?
        self.zero_grad()


class DeepWideNetwork(LanguageNetwork):
    def __init__(self, forward_evaluate, backward_evaluate):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate

        wide_sampler = PromptSampler(self.backward_evaluate, "wide_backward", num_samples=4)
        hidden_sampler = Sampler(self.backward_evaluate, "wide_backward", num_samples=4)  # hidden_backward
        agg_sampler = PromptSampler(self.backward_evaluate, "agg_backward", num_samples=4)

        self.wide = WideLayer(forward_evaluate, "wide_forward", 2, prompt_sampler=wide_sampler, hidden_sampler=hidden_sampler)
        self.agg = AggregationLayer(
            forward_evaluate,
            "agg_forward",
            prompt_sampler=agg_sampler,
            hidden_sampler=hidden_sampler,
            init="Therefore, the answer is:",
        )
        self.h = None
        self.outputs = None

    def forward(self, x):
        self.h = self.wide(x)
        self.outputs = self.agg(self.h)
        return self.outputs

        # any_out = self.outputs[0]
        # prompt_layers = any_out.prompt_layers_backward()
        # for l in prompt_layers:
        #     l.backward(losses, self.backward_evaluate)
        # self.zero_grad()

    def parameters(self):
        return self.wide.parameters() + self.agg.parameters()



class Sampler(ABC):

    def __init__(self, backward_evaluate, backward_template, num_samples=4):
        self.backward_evaluate = backward_evaluate
        self.backward_template = load_template(
            backward_template,
            template_directory="./templates"
        )
        self.num_samples = num_samples

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    # @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class PromptSampler(Sampler):

    def sample(self, prompts, losses, *args, **kwargs):
        """ Sample new prompts using the backward template.
            Returns a numpy array of shape (len(prompts), self.num_samples)
        """
        tpl_inputs = []
        for prompt in prompts:
            for _ in range(self.num_samples):
                tpl_inputs.append(
                    self.backward_template.render(
                        prompt=prompt, backward_infos=losses)
                )

        new_prompts = self.backward_evaluate(
            tpl_inputs,
            stop=self.backward_template.stop_tokens,
            **kwargs,
        )
        return np.asarray(new_prompts).reshape([len(prompts), self.num_samples])

        # local_losses_info = []
        # for _input, loss in zip(self.inputs, losses):
        #     concat_input = "\n".join((map(str, _input)))
        #     local_losses_info.append(Info(
        #         str(concat_input),
        #         str(loss.output),
        #         str(loss.target),
        #         loss.loss,
        #     ))

        # # propose new prompts using agg_backward
        # bwd_template = load_template(
        #     "agg_backward",
        #     template_directory="./templates"
        # )
        # tpl_inputs = [
        #     bwd_template.render(prompt=self.nodes[0].prompt, backward_infos=losses)
        #     for _ in range(n_samples)
        # ]

        # bwd_outputs = backward_evaluate(
        #     tpl_inputs,
        #     stop=bwd_template.stop_tokens,
        #     # **kwargs,
        # )


class HiddenSampler(Sampler):
    pass