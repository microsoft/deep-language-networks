import logging
from dataclasses import dataclass

import numpy as np
from typing import Union, List
from dln.operator import backward_evaluate
from dln.template import load_template, DLNTemplate

from dln.vi.utils import log_message


@dataclass
class Info:
    input: str = None
    output: str = None
    target: str = None
    loss: float = 0.0


class PromptSampler:
    def __init__(self, p_template="q_action_prompt:v3.5"):
        self.prompt_template = load_template(p_template)
        log_message("Prompt template:\n", f"{repr(self.prompt_template.template)}")
        log_message(
            "Message alternatives:\n", f"{self.prompt_template.message_alternatives}"
        )
        self.evaluate_func = backward_evaluate
        self.prompt_history = []

    @staticmethod
    def create(template):
        if "seq" in template:
            return SequentialPromptSampler()
        return PromptSampler(template)

    def sample_q_p(
        self,
        inputs: np.array,
        y: np.array,
        y_hat: np.array,
        losses: np.array,
        prompt: Union[str, List[str]],
        num_samples=1,
        held_out_half=False,
    ):
        """
        Args:
            inputs: input sequences
            y: target sequences
            y_hat: predicted sequences
            losses: losses for each sequence
            prompt: prompt to use for sampling
            num_samples: number of samples to generate
            held_out_half: if True, only use the first half of the data points for sampling prompts
        """
        infos = [
            Info(input=input_i, output=y_hat_i, target=y_i, loss=loss)
            for input_i, y_i, y_hat_i, loss in zip(inputs, y, y_hat, losses)
        ]
        while True:
            try:
                tpls = []
                for i in range(num_samples - 1):
                    template_infos = {}
                    if self.prompt_template.message_alternatives is None:
                        message = None
                    else:
                        message = self.prompt_template.message_alternatives[
                            i % len(self.prompt_template.message_alternatives)
                        ]

                    indices = np.random.permutation(np.arange(len(infos)))
                    if held_out_half:
                        infos_ = [infos[i] for i in indices[: len(infos) // 2]]
                    else:
                        infos_ = [infos[i] for i in indices]

                    template_infos["message"] = message
                    template_infos["backward_infos"] = infos_
                    template_infos["prompt"] = (
                        prompt[i % len(prompt)] if type(prompt) == list else prompt
                    )
                    tpls.append(self.prompt_template.render(**template_infos))

                log_message("Generating {} ~p proposals...".format(num_samples))
                new_prompts = self.evaluate_func(
                    tpls, stop=self.prompt_template.stop_tokens, n=1
                )
                log_message("DONE...")

                if type(prompt) == list:
                    prompts = np.array(prompt + list(new_prompts))
                else:
                    prompts = np.array([prompt] + list(new_prompts))
                return prompts
            except KeyboardInterrupt:
                break
            except:
                if len(infos) > 1:
                    infos = infos[1:]
                    logging.info("DROPPING A DATA POINT...")
                else:
                    error_message = (
                        "Still exeeding context length after shrinking backward_infos."
                    )
                    logging.info(error_message)
                    raise ValueError(error_message)


class SequentialPromptSampler(PromptSampler):
    def __init__(self):
        super().__init__(p_template="q_action_prompt_seq")

    def sample_q_p(
        self,
        inputs: np.array,
        y: np.array,
        y_hat: np.array,
        losses: np.array,
        prompt: str,
        num_samples=1,
        held_out_half=False,
    ):
        """
        Args:
            inputs: input sequences
            y: target sequences
            y_hat: predicted sequences
            losses: losses for each sequence
            prompt: prompt to use for sampling
            num_samples: number of samples to generate
            held_out_half: if True, only use the first half of the data points for sampling prompts
        """
        self.prompt_history.append(prompt)

        infos = [
            Info(input=input_i, output=y_hat_i, target=y_i, loss=loss)
            for input_i, y_i, y_hat_i, loss in zip(inputs, y, y_hat, losses)
        ]
        while True:
            try:
                tpls = []
                for i in range((num_samples - 1) // 3):
                    if self.prompt_template.message_alternatives is None:
                        message = None
                    else:
                        message = self.prompt_template.message_alternatives[
                            i % len(self.prompt_template.message_alternatives)
                        ]
                    indices = np.random.permutation(np.arange(len(infos)))
                    if held_out_half:
                        infos_ = [infos[i] for i in indices[: len(infos) // 2]]
                    else:
                        infos_ = [infos[i] for i in indices]
                    tpls.append(
                        self.prompt_template.render(
                            backward_infos=infos_,
                            prompt=self.prompt_history[-1],
                            message=message,
                        )
                    )

                log_message("Generating {} ~p proposals...".format(num_samples))

                prompts = self.evaluate_func(
                    tpls,
                    stop=self.prompt_template.stop_tokens,
                    n=1,
                )
                log_message("DONE...")

                # each prompt is prefix by 1., 2. and 3., so flatten the sequentially sampled prompts
                prompts_ = []
                for prompt_ in prompts:
                    sub_prompts_ = prompt_.split("\n")
                    sub_prompts_ = [sub_prompts_[0].strip()] + [
                        p_[2:].strip() for p_ in sub_prompts_[1:]
                    ]
                    sub_prompts_ = list(set(sub_prompts_))
                    prompts_.extend(sub_prompts_)

                prompts = np.array([prompt] + list(prompts_))
                return prompts
            except KeyboardInterrupt:
                break
            except:
                if len(infos) > 1:
                    infos = infos[1:]
                    logging.info("DROPPING A DATA POINT...")
                else:
                    error_message = (
                        "Still exeeding context length after shrinking backward_infos."
                    )
                    logging.info(error_message)
                    raise ValueError(error_message)


class PosteriorSampler:
    def __init__(self, q_template):
        self.q_templates = []
        
        if type(q_template) != DLNTemplate:
            for q_template in q_template.split("|"):
                self.q_templates.append(load_template(q_template))
        else:
            self.q_templates.append(q_template)

        for q_template in self.q_templates:
            log_message("Q template:", f"{repr(q_template.template)}")

        self.stop_tokens = self.q_templates[0].stop_tokens
        self.rng = np.random.RandomState(0)

    def sample_q_h(
        self,
        x: np.array,
        y: np.array,
        h: np.array,
        prompt: str,
        next_prompt: str,
        num_samples=1,
        strip_double_newlines=True,
        return_logprobs=False,
    ):
        """
        Sample a new hidden state from the posterior distribution.

        Args:
            x: inputs
            y: labels
            y_hat: model predictions for the forward pass
            h: hidden states for the forward pass
            task_description: task description if any
            prompt: prompt for the layer that generated h
            next_prompt: prompt for the layer above h
            forward_template: template for the forward pass that generated h
            num_samples: number of samples to generate
            strip_double_newlines: strip double new lines from the output samples
        Returns
            (batch_size, num_samples) array of hidden states
        """
        tpls = []

        for i, (x_i, h_i, y_i) in enumerate(zip(x, h, y)):
            for j in range(num_samples):
                # pick a template at random
                q_template = self.q_templates[
                    np.random.choice(np.arange(len(self.q_templates)))
                ]
                if q_template.message_alternatives is not None:
                    message = q_template.message_alternatives[
                        j % len(q_template.message_alternatives)
                    ]
                else:
                    message = None

                # pick another example in the set
                all_indices = list(np.arange(len(x)))
                all_indices.remove(i)
                source_example = self.rng.choice(all_indices)

                tpl = q_template.render(
                    input=x_i,
                    h=h_i,
                    source_x=x[source_example],
                    source_h=h[source_example],
                    prompt=prompt,
                    next_prompt=next_prompt,
                    y=y_i,
                    message=message,
                )

                # induce randomness
                tpls.append(tpl)

        assert len(
            tpls
        ), "If we are here, it means that either we resample hidden states, or that there are some errors."

        # this might happen when all memories are correct
        log_message("Q proposals: " + str(len(tpls)) + ", Q template:" + "\n" + tpls[0])
        log_message("Generating {} ~h proposals...".format(num_samples))

        sampled = backward_evaluate(
            tpls,
            stop=self.stop_tokens,
            n=1,
            return_logprobs=return_logprobs,
        )
        if return_logprobs:
            sampled, logprobs, lengths = zip(*sampled)
            logprobs = np.asarray(logprobs) / np.asarray(lengths)

        # strip any "\n\n" that might have been added
        if strip_double_newlines:
            sampled = [s.replace("\n\n", "\n") for s in sampled]

        sampled = np.asarray(sampled).reshape(x.shape[0], num_samples)
        assert sampled.shape[0] == x.shape[0]

        if return_logprobs:
            return sampled, logprobs.reshape(x.shape[0], num_samples)
        return sampled
