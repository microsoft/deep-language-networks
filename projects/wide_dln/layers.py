from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import numpy as np

from dln.operator import LLM
from dln.template import load_template


@dataclass
class LogProbs:
    logp_targets: np.ndarray
    distribution: np.ndarray


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

    def render_template(self, x):
        tpl_inputs = [
            self.forward_template.render(input=i, prompt=self.prompt)
            for i in x
        ]
        return tpl_inputs

    def __call__(self, x, **kwargs):
        tpl_inputs = self.render_template(x)
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
        prompt_scorer: "Scorer",
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
        self.prompt_scorer = prompt_scorer
        self.hidden_sampler = hidden_sampler
        self.hidden_scorer = hidden_sampler

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def prompts(self):
        return [n.prompt for n in self.nodes]

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        self.inputs = inputs
        outputs = [node(inputs, **kwargs) for node in self.nodes]
        self.outputs = outputs
        return np.asarray(outputs)

    def backward(self, losses):
        # select the one that maximizes the probability of the target
        # use the selected prompt to update all layer.nodes.prompt

        prompt_candidates = self.prompt_sampler(self.prompts, losses)
        best_prompts = self.prompt_scorer(prompt_candidates, losses, self)
        for n in self.nodes:
            n.prompt = best_prompts[n]
        return new_prompts  # return new outputs?

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
            l.backward(losses)
        self.zero_grad()


class DeepWideNetwork(LanguageNetwork):

    def __init__(self, forward_evaluate, backward_evaluate):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate

        wide_sampler = PromptSampler(self.backward_evaluate, "wide_backward", num_samples=4)
        hidden_sampler = PromptSampler(self.backward_evaluate, "wide_backward", num_samples=4)  # HiddenSampler hidden_backward
        agg_sampler = PromptSampler(self.backward_evaluate, "agg_backward", num_samples=4)
        prompt_scorer = LogProbsScorer(self.forward_evaluate)

        self.wide = WideLayer(
            forward_evaluate,
            "wide_forward",
            2,
            prompt_sampler=wide_sampler,
            prompt_scorer=prompt_scorer,
            hidden_sampler=hidden_sampler,
        )
        self.agg = AggregationLayer(
            forward_evaluate,
            "agg_forward",
            prompt_sampler=agg_sampler,
            prompt_scorer=prompt_scorer,
            hidden_sampler=hidden_sampler,
            init="Therefore, the answer is:",
        )
        self.h = None
        self.outputs = None

    def forward(self, x):
        self.h = self.wide(x)
        self.outputs = self.agg(self.h)
        return self.outputs

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

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class PromptSampler(Sampler):

    def sample(self, prompts, losses, **kwargs):
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


class Scorer(ABC):

    def __init__(self, forward_evaluate, eval_kwargs=None):
        self.forward_evaluate = forward_evaluate
        self.eval_kwargs = eval_kwargs or {}

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    @abstractmethod
    def score(self, *args, **kwargs):
        pass


class LogProbsScorer(Scorer):

    def __init__(self, forward_evaluate, eval_kwargs=None):
        eval_kwargs = {
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "return_logprobs": True,
            "raw_logprobs": True,
        }
        super().__init__(forward_evaluate, eval_kwargs)

    def _eval_context(self, prompts, layer):
        fwd_rendered_template = []
        for node, candidates in zip(layer.nodes, prompts):
            for prompt_candidate in candidates:
                for i in layer.inputs:
                    fwd_rendered = node.forward_template.render(input=i, prompt=prompt_candidate)
                    fwd_rendered_template.append(fwd_rendered.replace('[END]', ''))  # TODO: clean prompt when generating, not here
        return fwd_rendered_template

    def _forward_unique_evals(self, eval_batch):
        # there might be doubles in the eval_batch, so we need to only perform unique evals
        unique_keys = list(set(eval_batch))
        unique_keys_to_positions = {key: i for i, key in enumerate(unique_keys)}
        unique_eval_results = self.forward_evaluate(
            unique_keys,
            async_generation=True,
            **self.eval_kwargs,
        )
        # get the results in the same order as the eval_batch
        eval_results = []
        for eval_key in eval_batch:
            eval_results.append(unique_eval_results[unique_keys_to_positions[eval_key]])
        return eval_results

    def _get_logprobs_results(self, contexts, eval_results):
        log_probs = [e[1] for e in eval_results]
        output_logprobs = []
        context_logprobs = []

        burn_in = 0
        for context, token_log_probs in zip(contexts, log_probs):
            num_tokens_prompt = len(self.forward_evaluate.encoder.encode(context))
            target_log_probs = token_log_probs[num_tokens_prompt + burn_in:]
            context_log_probs = token_log_probs[1:num_tokens_prompt]

            if len(target_log_probs) == 0:
                output_logprobs.append("empty")
            else:
                output_logprobs.append(
                    sum(target_log_probs) / (len(target_log_probs) + 1e-5)
                )

            context_logprobs.append(
                sum(context_log_probs) / (len(context_log_probs) + 1e-5)
            )

        non_empty = [o for o in output_logprobs if o != "empty"]
        if len(non_empty) == 0:
            min = 0
        else:
            min = np.min(non_empty)
        output_logprobs = [o if o != "empty" else min for o in output_logprobs]
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))  # TODO: reshape?


    def score(self, prompts_candidates, losses, layer, **kwargs):
        _, num_candidates = prompts_candidates.shape
        contexts = self._eval_context(prompts_candidates, layer)
        eval_batch = [f"{c}\n{l.target}" for c, l in zip(contexts, losses * num_candidates)]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)
        scores = logprobs_results.logp_targets.reshape(
            len(layer.inputs), num_candidates
        ).sum(1, keepdims=True)
        best_index = np.argmax(scores)
        return prompts_candidates[:, best_index]
