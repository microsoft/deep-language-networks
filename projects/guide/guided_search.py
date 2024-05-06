from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List, Optional, Tuple

from dln.operator import GPT
from dln.template import load_template
from dln.postprocessing import remove_extra_spaces


class Rating(Enum):
    GOOD = 1
    BAD = -1
    NEUTRAL = 0

    @classmethod
    def names(cls):
        return [member.name.capitalize() for member in cls]

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    @classmethod
    def icons(cls, i):
        return {
            1: "👍",
            0: "Skip",
            -1: "👎"
        }[i]


@dataclass
class ExampleOutput:
    prompt: str
    example: str
    output: str
    rating: Optional[Rating]
    feedback: str


class GuidedSearch:

    def __init__(
        self,
        forward_template="classify_forward",
        backward_template="guided_backward:v3.0",
        consolidate_template="guided_backward:v5.0",
    ):
        self.forward_template = load_template(forward_template, "./templates/")
        self.backward_template = load_template(backward_template, "./templates/")
        self.consolidate_template = load_template(consolidate_template, "./templates/")

        self._openai_setup()
        gpt_35t = "gpt-35-turbo" if os.getenv("OPENAI_API_TYPE") == 'azure' else "gpt-3.5-turbo"

        self._avaliable_models = {
            "GPT-3.5-Turbo": gpt_35t,
            "GPT-4-Turbo": "gpt-4-turbo",
            "GPT-4": "gpt-4",
        }

        self.fwd_config = {
            "temperature": 0.0,
            "max_tokens": 200,
            "stop": None,
        }
        self.fwd_model = GPT(gpt_35t)

        self.bwd_config = {
            "temperature": 0.7,
            "max_tokens": 200,
        }
        self.bwd_model = GPT(gpt_35t)

    @staticmethod
    def _openai_setup():
        import os  # azure bug workaround
        if os.environ.get("OPENAI_API_TYPE") == 'azure':
            import openai
            openai.api_version = os.environ.get("OPENAI_API_VERSION")

    @property
    def available_models(self) -> List[str]:
        return list(self._avaliable_models.keys())

    def update_fwd_configs(self, model: str, temperature: float, max_tokens: int) -> None:
        model_id = self._avaliable_models[model]
        if model_id != self.fwd_model.engine:
            self.fwd_model = GPT(model_id)
        self.fwd_config["temperature"] = temperature
        self.fwd_config["max_tokens"] = max_tokens

    def update_bwd_configs(self, temperature: float, max_tokens: int) -> None:
        self.bwd_config["temperature"] = temperature
        self.bwd_config["max_tokens"] = max_tokens

    @staticmethod
    def _remove_extra_spaces(sentences: List[str]) -> List[str]:
        return [remove_extra_spaces(s, remove_new_line=True) for s in sentences]

    def inference(self, prompt: str, examples: List[str]) -> List[str]:
        """
        Given a prompt and a list of examples, returns a list of predicted outputs.

        Args:
            prompt (str): The prompt to generate predictions for.
            examples (List[str]): A list of examples to use for prediction.

        Returns:
            List[str]: A list of predicted outputs.
        """
        fwd_examples = [
            self.forward_template.render(prompt=prompt, input=example)
            for example in examples
        ]
        outputs = self.fwd_model(fwd_examples, async_generation=True, **self.fwd_config)
        return self._remove_extra_spaces(outputs)

    def inference_per_example(self, prompts: List[str], example: str) -> List[str]:
        """
        Given a list of prompts and an example, generates a list of strings representing the model's inference
        for each prompt applied to the example.

        Args:
            prompts (List[str]): A list of prompts to be applied to the example.
            example (str): A string representing the example to be used for inference.

        Returns:
            List[str]: A list of strings representing the model's inference for each prompt applied to the example.
        """
        fwd_examples = [
            self.forward_template.render(prompt=prompt, input=example)
            for prompt in prompts
        ]
        outputs = self.fwd_model(fwd_examples, async_generation=True, **self.fwd_config)
        return self._remove_extra_spaces(outputs)

    def prompt_proposal(
        self,
        current_prompt: str,
        example_outputs: List[ExampleOutput],
        feedback: Optional[str] = "",
        num_samples: int = 3,
    ) -> List[str]:
        """
        Generate a list of prompt candidates based on the current prompt,
        examples of examples outputs, and optional feedback from the user.

        Args:
            current_prompt (str): The current prompt to be extended or modified.
            examples_outputs (List[OutputScore]): A list of OutputScores.
            feedback (Optional[str]): Optional feedback from the user on the generated outputs to guide the prompt generation.

        Returns:
            List[str]: A list of prompt candidates generated based on the examples_outputs and feedback.
        """
        bwd_prompt = self.backward_template.render(
            prompt=current_prompt,
            example_outputs=example_outputs,
            feedback=feedback,
        )
        bwd_prompts = [bwd_prompt] * num_samples
        prompt_proposals = self.bwd_model(
            bwd_prompts,
            async_generation=True,
            **self.bwd_config
        )

        # If we are missing prompt proposals, we resample while increasing the temperature.
        bwd_config = self.bwd_config.copy()
        while len(set(prompt_proposals)) < num_samples:
            bwd_config["temperature"] += 0.1
            print(f"{len(set(prompt_proposals))} unique meta-prompts out of {num_samples} samples, resampling with a higher temperature {bwd_config['temperature']}")
            prompt_proposals += self.bwd_model(
                bwd_prompts,
                async_generation=True,
                **bwd_config
            )

        # Order-preserving set of prompt proposals.
        prompt_proposals = list(dict.fromkeys(prompt_proposals))
        return prompt_proposals[:num_samples]

    def select_prompt(self, prompt_candidates: Dict[str, List[ExampleOutput]]) -> Tuple[str, List[ExampleOutput]]:
        """
        Selects the best prompt from a dictionary of prompt candidates based on user feedback.
        The selection is based on the prompt with the highest sum of ratings.

        Args:
            prompt_candidates (Dict[str, OutputScore]): A dictionary where the keys are prompt strings,
                and the values are tuples containing the input, output, and a rating.

        Returns:
            str: The prompt string with the most user votes.
        """
        prompt_votes = {}
        for prompt, output_scores in prompt_candidates.items():
            for output_score in output_scores:
                prompt_votes[prompt] = prompt_votes.get(prompt, 0) + output_score.rating.value
        if not prompt_candidates or not sum(prompt_votes.values()):
            return None, None
        selected_prompt = max(prompt_votes, key=prompt_votes.get)
        return selected_prompt, prompt_candidates[selected_prompt]

    def consolidate_prompt(self, prompt_candidates: Dict[str, List[ExampleOutput]]) -> str:
        """
        Consolidates the prompt candidates into a single prompt using LLMs.

        Args:
            prompt_candidates (Dict[str, OutputScore]): A dictionary where the keys are prompt strings,
                and the values are tuples containing the input, output, and a boolean indicating whether the user
                prefered the output compared to the other outputs.
        Returns:
            str: The consolidated prompt.
        """
        flat_example_outputs = []
        for example_outputs in prompt_candidates.values():
            flat_example_outputs.extend(example_outputs)
        consolidate_prompt = self.consolidate_template.render(example_outputs=flat_example_outputs)
        bwd_config = self.bwd_config.copy()
        bwd_config["temperature"] = 0.0
        outputs = self.bwd_model(consolidate_prompt, async_generation=True, **bwd_config)
        consolidated_prompt = self._remove_extra_spaces(outputs)[0]
        return consolidated_prompt


class GuidedSearchController:

    def __init__(self) -> None:
        self.gs = GuidedSearch()
        self.meta_prompt: str = ""
        self.feedback: str = ""
        self.num_candidates: int = 3
        self.examples: List[str] = []  # inputs
        self.example_outputs: List[ExampleOutput] = []
        self.optimization_step: int = 1
        self.history: Dict[str: List[ExampleOutput]] = {}
        self.prompt_candidates_outputs: Dict[str: List[ExampleOutput]] = {}

    def setup(self, meta_prompt: str, inputs_text: str) -> None:
        self.meta_prompt = meta_prompt
        # Split the input examples by new line.
        self.examples = inputs_text.split('\n')
        self.inference_step()

    def update_bwd_configs(
        self,
        num_candidates: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.num_candidates = num_candidates
        self.gs.update_bwd_configs(temperature, max_tokens)

    @property
    def available_models(self) -> List[str]:
        return self.gs.available_models

    def update_fwd_configs(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.gs.update_fwd_configs(model, temperature, max_tokens)

    @property
    def bwd_temperature(self):
        return self.gs.bwd_config["temperature"]

    @property
    def bwd_max_tokens(self):
        return self.gs.bwd_config["max_tokens"]

    @property
    def fwd_temperature(self):
        return self.gs.fwd_config["temperature"]

    @property
    def fwd_max_tokens(self):
        return self.gs.fwd_config["max_tokens"]

    def inference_step(self):
        outputs = self.gs.inference(self.meta_prompt, self.examples)
        self.example_outputs = [
            ExampleOutput(self.meta_prompt, example, output, None, "")
            for example, output in zip(self.examples, outputs)
        ]
        self.history.update({self.meta_prompt: self.example_outputs})

    def prompt_proposal_step(self):
        prompt_candidates = self.gs.prompt_proposal(
            self.meta_prompt,
            self.example_outputs,
            self.feedback,
            num_samples=self.num_candidates,
        )
        self.prompt_candidates_outputs = {
            candidate: []
            for candidate in prompt_candidates
        }

    def inference_candidates_per_example(self):
        prompt_candidates = self.prompt_candidates_outputs.keys()
        for example in self.examples:
            outputs = self.gs.inference_per_example(prompt_candidates, example)
            for prompt_candidate, output in zip(prompt_candidates, outputs):
                self.prompt_candidates_outputs[prompt_candidate].append(
                    ExampleOutput(prompt_candidate, example, output, None, "")
                )

    def find_outputs_by_example(self, example: str) -> List[str]:
        outputs = []
        for example_outputs in self.prompt_candidates_outputs.values():
            for example_output in example_outputs:
                if example_output.example == example:
                    outputs.append(example_output.output)

        return outputs

    def update_ratings(self, selected_outputs):
        for j, output_example in enumerate(self.prompt_candidates_outputs.values()):
            for i, output in enumerate(output_example):
                rating, feedback = selected_outputs[(j, i)]
                output.rating = Rating[rating.upper()] if isinstance(rating, str) else Rating(rating)
                output.feedback = feedback

    def select_prompt(self, selected_outputs):
        self.update_ratings(selected_outputs)
        selected_prompt, output_score = self.gs.select_prompt(self.prompt_candidates_outputs)
        if selected_prompt is not None:
            # Update the meta prompt and use example outputs cache instead of self.inference_step().
            self.meta_prompt = selected_prompt
            self.example_outputs = output_score
        self.history.update({self.meta_prompt: self.example_outputs})
        self.optimization_step += 1
        self.prompt_candidates_outputs = {}

    def consolidate_prompts(self, selected_outputs):
        self.update_ratings(selected_outputs)
        consolidated_prompt = self.gs.consolidate_prompt(self.prompt_candidates_outputs)
        self.meta_prompt = consolidated_prompt
        self.inference_step()
        self.optimization_step += 1
        self.prompt_candidates_outputs = {}


if __name__ == "__main__":
    gs = GuidedSearch()
    prompts = ["What is the result of", "Resolve the following equation", "Calculate the following"]
    examples = ["1 + 1", "2 + 2", "3 + 3"]
    outputs = gs.inference(prompts[0], examples)
    print(outputs)
    outputs = gs.inference_per_example(prompts, examples[0])
    print(outputs)

    prompt_candidates = {
        "What is the result of": [
            ExampleOutput("What is the result of", "1 + 1", "2", True),
            ExampleOutput("What is the result of", "2 + 2", "4", False),
            ExampleOutput("What is the result of", "3 + 3", "6", True),
        ],
        "Resolve the following equation": [
            ExampleOutput("Resolve the following equation", "1 + 1", "3", False),
            ExampleOutput("Resolve the following equation", "2 + 2", "4", True),
            ExampleOutput("Resolve the following equation", "3 + 3", "5", False),
        ],
        "Calculate the following": [
            ExampleOutput("Resolve the following equation", "1 + 1", "2", False),
            ExampleOutput("Resolve the following equation", "2 + 2", "3", False),
            ExampleOutput("Resolve the following equation", "3 + 3", "4", False),
        ]
    }
    selected_prompt, output_score = gs.select_prompt(prompt_candidates)
    print(selected_prompt)

    consolidated_prompt = gs.consolidate_prompt(prompt_candidates)
    print("Consolidated Prompt: ")
    print(consolidated_prompt)
    print(output_score)

    feedback = "Paraphrase the question"
    prompt_proposals = gs.prompt_proposal(selected_prompt, output_score, feedback)
    print(prompt_proposals)

    gsc = GuidedSearchController()
    gsc.setup("What is the result of", "1 + 1\n2 + 2\n3 + 3")
    print(f"Meta Prompt: {gsc.meta_prompt}")
    print(f"History: {gsc.history}")

    gsc.inference_step()
    print(f"Meta Prompt: {gsc.meta_prompt}")
    print(f"History: {gsc.history}")

    gsc.feedback = feedback
    gsc.prompt_proposal_step()
    gsc.inference_candidates_per_example()
    print(f"Candidates outputs: {gsc.prompt_candidates_outputs}")