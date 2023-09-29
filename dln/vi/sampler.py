import logging
from dataclasses import dataclass

import numpy as np
from dln.operator import LLM
from dln.template import load_template

from dln.vi.utils import log_message


@dataclass
class Info:
    input: str = None
    output: str = None
    target: str = None
    loss: float = 0.0


class PromptSampler:
    def __init__(self, backward_evaluate: LLM, p_template: str = "q_action_prompt:v3.5"):
        self.prompt_template = load_template(p_template)
        log_message("Prompt template:\n", f"{repr(self.prompt_template.template)}")
        log_message(
            "Message alternatives:\n", f"{self.prompt_template.message_alternatives}"
        )
        self.evaluate_func = backward_evaluate

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
        infos = [
            Info(input=input_i, output=y_hat_i, target=y_i, loss=loss)
            for input_i, y_i, y_hat_i, loss in zip(inputs, y, y_hat, losses)
        ]
        while True:
            try:
                tpls = []
                for i in range(num_samples - 1):
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
                            prompt=prompt,
                            message=message,
                        )
                    )

                log_message("Prompt Sampler:", tpls[-1])
                log_message("Generating {} ~p proposals...".format(num_samples))

                prompts = self.evaluate_func(
                    tpls,
                    stop=self.prompt_template.stop_tokens,
                    n=1,
                    async_generation=True,
                )
                log_message("DONE...")

                prompts = np.array([prompt] + list(prompts))
                return prompts
            except KeyboardInterrupt:
                break
            except:
                if len(infos) > 1:
                    infos = infos[1:]
                    logging.info("DROPPING A DATA POINT...")
                else:
                    error_message = "Still exeeding context length after shrinking backward_infos."
                    logging.info(
                        error_message
                    )
                    raise ValueError(error_message)


class PosteriorSampler:
    def __init__(self, backward_evaluate: LLM, q_template: str):
        self.q_templates = []
        for q_template in q_template.split("|"):
            self.q_templates.append(load_template(q_template))
        for q_template in self.q_templates:
            log_message("Q template:", f"{repr(q_template.template)}")
        self.stop_tokens = self.q_templates[0].stop_tokens
        self.evaluate_func = backward_evaluate

    def sample_q_h(
        self,
        x: np.array,
        y: np.array,
        h: np.array,
        prompt: str,
        next_prompt: str,
        num_samples=1,
        strip_double_newlines=True,
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

        for x_i, h_i, y_i in zip(x, h, y):
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
                tpl = q_template.render(
                    input=x_i,
                    h=h_i,
                    prompt=prompt,
                    next_prompt=next_prompt,
                    y=y_i,
                    message=message,
                )
                tpls.append(tpl)

        # WATCH OUT: we only use max_tokens=128
        max_tokens = 256
        assert len(
            tpls
        ), "If we are here, it means that either we resample hidden states, or that there are some errors."

        # this might happen when all memories are correct
        log_message("Q proposals: " + str(len(tpls)) + ", Q template:" + "\n" + tpls[0])
        log_message(
            "Generating {} ~h proposals... max_tokens={}".format(
                num_samples, max_tokens
            )
        )
        sampled = self.evaluate_func(
            tpls,
            stop=self.stop_tokens,
            n=1,
            max_tokens=max_tokens,
            async_generation=True,
        )

        # strip any "\n\n" that might have been added
        if strip_double_newlines:
            sampled = [s.replace("\n\n", "\n") for s in sampled]

        sampled = np.asarray(sampled).reshape(x.shape[0], num_samples)
        assert sampled.shape[0] == x.shape[0]
        return sampled
