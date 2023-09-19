import numpy as np
from abc import ABC, abstractmethod
from dln.template import DLNTemplate
import logging

from dataclasses import dataclass

from engine.network import NetworkNode
from engine.scorer import VIScorer
from engine.template import load_template
from engine.layers import LanguageLayer


@dataclass
class Info:
    input: str = None
    output: str = None
    target: str = None
    loss: float = 0.0


class PromptSampler(ABC):
    def __init__(self):
        # dependency injection!
        self._base_layer: NetworkNode = None

    @property
    def base_layer(self):
        if self._base_layer is None:
            raise ValueError("Base layer not registered.")
        return self._base_layer

    @property
    def backward_lm(self):
        return self.base_layer.backward_lm

    def register_base_layer(self, base_layer: LanguageLayer):
        """Register the base layer this sampler is attached to.

        Args:
            base_layer (LanguageLayer): The base layer this sampler is attached to.
        """
        self._base_layer = base_layer

    @abstractmethod
    def rewrite_prompts(
        self,
        y: np.array,
        inputs: np.array = None,
        losses: np.ndarray = None,
        num_samples: int = 1,
    ):
        pass


class HiddenSampler(ABC):
    def __init__(self):
        # dependency injection!
        self._base_layer: NetworkNode = None

    @property
    def base_layer(self):
        if self._base_layer is None:
            raise ValueError("Base layer not registered.")
        return self._base_layer

    @property
    def backward_lm(self):
        return self.base_layer.backward_lm

    def register_base_layer(self, base_layer: LanguageLayer):
        """Register the base layer this sampler is attached to.

        Args:
            base_layer (LanguageLayer): The base layer this sampler is attached to.
        """
        self._base_layer = base_layer

    @abstractmethod
    def rewrite_inputs(
        self,
        y: np.array,
        num_samples: int = 1,
    ):
        pass


@dataclass
class InputsRewrites:
    inputs: np.array
    inputs_logps: np.array = None


class BackpropHiddenSampler(HiddenSampler):
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

    def __init__(self):
        super().__init__()

        self.backwards_template = self.get_backwards_inputs_template()

    def rewrite_inputs(self, y: np.array, num_samples: int = 1):
        # now re-write the inputs
        repeated_inputs = [
            input for input in self.base_layer.inputs_cache for _ in range(num_samples)
        ]
        repeated_targets = [target for target in y for _ in range(num_samples)]
        repeated_inputs = [
            self.get_backwards_template.render(
                prompt=self.base_layer.weight,
                input=input,
                y=target,
            )
            for input, target in zip(repeated_inputs, repeated_targets)
        ]
        new_inputs = self.backward_lm.generate(
            repeated_inputs,
            n=1,
        )
        return InputsRewrites(
            inputs=np.asarray(new_inputs).reshape(-1, num_samples),
        )


class MultiActionPromptSampler(PromptSampler):
    def __init__(self, use_memory=True):
        super().__init__()

        self.use_memory = use_memory
        if use_memory:
            self.prompt_template = load_template("q_action_prompt_mem:v3.5")
        else:
            self.prompt_template = load_template("q_action_prompt:v3.5")
        self.memory = {}

    def rewrite_prompts(
        self,
        y: np.array,
        inputs: np.array = None,
        losses: np.ndarray = None,
        num_samples: int = 1,
    ):
        if self.use_memory:
            prompts = getattr(self.base_layer, "candidate_prompts", [])
            prompt_weights = getattr(self.base_layer, "candidate_prompts_scores", [])
            self.memory.update({p: w for p, w in zip(prompts, prompt_weights)})

        if len(self.memory) == 0:
            prompt_memories = [(self.base_layer.weight, 1.0)]
        else:
            prompt_memories = sorted(list(self.memory.items()), key=lambda x: -x[1])[
                :10
            ][::-1]

        infos = [
            Info(input=input_i, output=y_hat_i, target=y_i, loss=loss)
            for input_i, y_i, y_hat_i, loss in zip(
                inputs if inputs is not None else self.base_layer.inputs_cache,
                y,
                self.base_layer.outputs_cache,
                losses,
            )
        ]

        while True:
            try:
                tpls = []
                for i in range(num_samples):
                    template_infos = {}
                    if self.prompt_template.message_alternatives is None:
                        message = None
                    else:
                        message = self.prompt_template.message_alternatives[
                            i % len(self.prompt_template.message_alternatives)
                        ]

                    indices = np.random.permutation(np.arange(len(infos)))
                    infos_ = [infos[i] for i in indices]

                    template_infos["message"] = message
                    template_infos["backward_infos"] = infos_
                    template_infos["prompt"] = self.base_layer.weight
                    template_infos["prompt_memories"] = prompt_memories
                    tpls.append(self.prompt_template.render(**template_infos))

                logging.info(tpls[0])
                logging.debug("Generating {} ~p proposals...".format(num_samples))

                new_prompts = self.backward_lm.generate(
                    tpls, stop=self.prompt_template.stop_tokens, n=1
                )

                prompts = np.array([self.base_layer.weight] + list(new_prompts))
                return prompts

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.warn("Gotten exception {}".format(e))

                if len(infos) > 1:
                    infos = infos[1:]
                else:
                    error_message = (
                        "Still exeeding context length after shrinking backward_infos."
                    )
                    logging.info(error_message)
                    raise ValueError(error_message)


class PriorHiddenSampler(HiddenSampler):
    def rewrite_inputs(
        self,
        y: np.array,
        num_samples: int = 1,
    ):
        # now re-write the inputs
        previous_node = self.base_layer.input_nodes[0]
        repeated_inputs = [
            input for input in previous_node.inputs_cache for _ in range(num_samples)
        ]
        repeated_inputs = previous_node.instantiate_template(repeated_inputs)

        new_inputs = self.backward_lm.generate(
            repeated_inputs,
            n=1,
            stop=previous_node.forward_template.stop_tokens,
            return_logprobs=self.backward_lm.has_log_probs,
        )

        if self.backward_lm.has_log_probs:
            new_inputs, new_logps, new_lengths = zip(*new_inputs)
            input_logps = (np.asarray(new_logps) / np.asarray(new_lengths)).reshape(
                -1, num_samples
            )
        else:
            input_logps = None

        return InputsRewrites(
            inputs=np.asarray(new_inputs).reshape(-1, num_samples),
            inputs_logps=input_logps,
        )

