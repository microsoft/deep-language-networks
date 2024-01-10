from abc import ABC, abstractmethod
from contextlib import contextmanager
import re
from typing import Dict, List, Optional, Union
import asyncio
import numpy as np
import openai
import logging
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from termcolor import colored
import yaml


openai.util.logger.setLevel(logging.WARNING)


def _retry_request(min_wait=4, max_wait=10, max_attempts=100):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )


def _parse_openai_response(
    response,
    return_logprobs=False,
    raw_logprobs=False,
    top_logprobs=False,
    **kwargs,
):
    output = []
    nlls = []
    lengths = []
    for response in response["choices"]:
        output.append(response["text"].strip())
        if raw_logprobs:
            nlls.append(response["logprobs"]["token_logprobs"])
            lengths.append(response["logprobs"]["tokens"])
        elif top_logprobs:
            nlls.append(response["logprobs"]["top_logprobs"])
            lengths.append(response["logprobs"]["tokens"])
        else:
            if "token_logprobs" in response["logprobs"]:
                nlls.append(sum(response["logprobs"]["token_logprobs"]))
                lengths.append(len(response["logprobs"]["token_logprobs"]))
            else:
                nlls.append(-np.inf)
                lengths.append(1)

    if return_logprobs:
        output = list(zip(output, nlls, lengths))
    return output


class LLM(ABC):

    def __init__(self, model_name: str, seed: Optional[int] = None, **generation_options):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed) if seed is not None else None
        self.generation_options = generation_options
        self.engine = model_name
        self.total_cost = 0.0

    def __call__(self, inputs: Union[List[str], str], **kwargs) -> List[str]:
        """Generate outputs for the given inputs. Use this method instead of calling _generate directly.
        In addition to calling LLM._generate, this method generates a random seed per request
        if the LLM was instantiated with a seed, and calculates the cost of the generation.
        If a seed is provided in the kwargs, it will be used instead of generating a random one.
        """
        is_echo_enabled = kwargs.get("echo") or self.generation_options.get("echo")
        if not is_echo_enabled:
            self.compute_cost(inputs)
        # if LLM has a seed generator, and no seed is provided, generate a random seed
        if kwargs.get("seed") is None and self.rng is not None:
            kwargs["seed"] = self._gen_random_seed()
        outputs = self._generate(inputs, **kwargs)

        if kwargs.get("return_logprobs"):
            self.compute_cost([out[0] for out in outputs])
        else:
            self.compute_cost(outputs)
        return outputs

    @abstractmethod
    def _generate(self, inputs: Union[List[str], str], **kwargs) -> List[str]:
        """Generate outputs for the given inputs. Do not call this method directly,
        since it does not generate a random seed or calculate the cost of the generation.
        Use llm_instance(inputs) instead. Refer to __call__ method for more details.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, string: str) -> List[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def has_logprobs(self) -> bool:
        raise NotImplementedError

    def compute_cost(self, inputs: List[str]) -> float:
        self.total_cost += np.sum(list([len(self.encode(input)) for input in inputs]))

    def _gen_random_seed(self):
        return self.rng.randint(0, 10000)


class GPT(LLM):

    CHAT_COMPLETION_MODELS = [
        "gpt-35-turbo",  # azure
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-0613",
    ]

    COMPLETION_MODELS = [
        "gpt-35-turbo-instruct",  # azure
        "gpt-3.5-turbo-instruct",
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    ]

    AVAILABLE_MODELS = CHAT_COMPLETION_MODELS + COMPLETION_MODELS
    LOGPROBS_MODELS = COMPLETION_MODELS.copy()

    def __init__(self, model_name: str = "text-davinci-003", **generation_options):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"GPT model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )
        super().__init__(model_name, **generation_options)
        engine_for_encoder = self.engine.replace("gpt-35", "gpt-3.5")
        self.encoder = instantiate_tokenizer(engine_for_encoder)
        openai.api_version = os.environ.get('OPENAI_API_VERSION')
        self._has_logprobs = self.engine in self.LOGPROBS_MODELS

    def encode(self, string: str) -> List[int]:
        return self.encoder.encode(string)

    @property
    def has_logprobs(self) -> bool:
        return self._has_logprobs

    @staticmethod
    def _log_filtering_error_message(error_message, prompt):
        error_message = (
            f"InvalidRequestError, most likely due to content filtering. "
            f"Prompt: {prompt}. ErrorMessage: {error_message}"
        )
        logging.warning(error_message)
        print(colored(error_message, "red"))

    @_retry_request(min_wait=4, max_wait=10, max_attempts=100)
    async def _aget_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            kwargs["deployment_id"] = self.engine
        else:
            kwargs["model"] = self.engine
        try:
            response = await openai.ChatCompletion.acreate(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except openai.InvalidRequestError as e:
            self._log_filtering_error_message(e, prompt)
            raise e

        if "content" not in response["choices"][0]["message"]:
            return ""

        output = response["choices"][0]["message"]["content"].strip()
        return output

    @_retry_request(min_wait=4, max_wait=10, max_attempts=500)
    def _get_completion_response(
        self,
        prompt_batch,
        return_logprobs=False,
        raw_logprobs=False,
        top_logprobs=False,
        **kwargs,
    ):
        """
        prompting gpt-3 via openai api
        now batching only works for completion, not on chat
        """
        logging.debug(kwargs)
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt_batch,
                logprobs=top_logprobs or 1,
                **kwargs,
            )
        except openai.InvalidRequestError as e:
            # Retry one by one to find out which prompt is causing the error for debugging
            try:
                for prompt in prompt_batch:
                    _ = openai.Completion.create(
                        engine=self.engine,
                        prompt=prompt,
                        logprobs=top_logprobs or 1,
                        **kwargs,
                    )
            except openai.InvalidRequestError as err:
                self._log_filtering_error_message(err, prompt)
            raise e

        return _parse_openai_response(response, return_logprobs, raw_logprobs, top_logprobs)

    async def _gather_chat_response(self, inputs, **generation_options):
        outputs = await asyncio.gather(
            *[
                self._aget_chat_completion_response(_input, **generation_options)
                for _input in inputs
            ]
        )
        return outputs

    def _mini_batch(self, inputs, batch_size=20):
        input_length = len(inputs)
        num_batches = input_length // batch_size + (
            1 if input_length % batch_size > 0 else 0
        )
        for i in range(num_batches):
            input_batch = inputs[batch_size * i : batch_size * (i + 1)]
            yield input_batch

    def _generate(
        self,
        inputs: Union[List[str], str],
        async_generation: bool = True,
        batch_size: int = 20,
        **kwargs,
    ) -> List[str]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)

        if "return_logprobs" in generation_options and not self.has_logprobs:
            logging.warning(
                f"return_logprobs is not supported for model {self.engine}"
            )
            del generation_options["return_logprobs"]

        if self.engine in self.CHAT_COMPLETION_MODELS:
            if async_generation is True:
                # async call api, devide to mini batches to avoid call rate limit
                outputs = []
                for input_batch in self._mini_batch(inputs, batch_size=batch_size):
                    outputs_batch = asyncio.run(
                        self._gather_chat_response(input_batch, **generation_options)
                    )
                    outputs = outputs + outputs_batch
            else:
                # call api one by one
                outputs = [
                    asyncio.run(
                        self._aget_chat_completion_response(_input, **generation_options)
                    )
                    for _input in inputs
                ]
        else:
            # completion_models, devide to mini batches (max batch size = 20 according to openai)
            outputs = []
            for input_batch in self._mini_batch(inputs, batch_size=batch_size):
                outputs_batch = self._get_completion_response(
                    input_batch, **generation_options
                )
                outputs = outputs + outputs_batch
        return outputs


class VLLM(LLM):

    def __init__(self, model_name: str, **generation_options):
        super().__init__(model_name, **generation_options)
        self.encoder = instantiate_tokenizer(model_name)

    @_retry_request(min_wait=1, max_wait=1, max_attempts=100)
    async def _aget_vllm_response(self, input, **kwargs):
        response = await openai.Completion.acreate(
            model=self.engine,
            prompt=input,
            logprobs=kwargs.get("top_logprobs") or 1,
            **kwargs,
        )
        return _parse_openai_response(response, **kwargs)[0]

    async def _gather_vllm_response(self, inputs, **kwargs):
        outputs = await asyncio.gather(
            *[
                self._aget_vllm_response(_input, **kwargs)
                for _input in inputs
            ]
        )
        return outputs

    def _generate(
        self,
        inputs: Union[List[str], str],
        async_generation: bool = True,
        **kwargs
    ) -> List[str]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)
        if async_generation:
            outputs = asyncio.run(
                self._gather_vllm_response(inputs, **generation_options)
            )
        else:
            outputs = [
                asyncio.run(
                    self._aget_vllm_response(_input, **generation_options)
                )
                for _input in inputs
            ]
        return outputs

    def encode(self, string: str) -> List[int]:
        return self.encoder.encode(string)

    @property
    def has_logprobs(self) -> bool:
        return True


def instantiate_tokenizer(model_name: str):
    if model_name in GPT.AVAILABLE_MODELS:
        import tiktoken
        encoder = tiktoken.encoding_for_model(model_name)
    else:
        from transformers import AutoTokenizer
        if model_name.startswith("/"):
            pretrained_path = os.getenv("TOKENIZER_PATH")
        else:
            pretrained_path = model_name
        encoder = AutoTokenizer.from_pretrained(pretrained_path)
    return encoder


class LLMRegistry:

    def __init__(self, config=None):
        self.models : Dict[str, LLM] = {}
        if config is not None:
            self._load_from_configs(config)

    def register(self, model_name: str, model_type: str = None, **generation_options) -> LLM:
        """Register a single model to the LLMRegistry.
        Args:
            model_name: how you refer to the model, for example: gpt-3.
            model_type: the api model name, for example: text-davinci-003. If not provided, use model_name as default.
            **generation_options: generation options, for example: api_key, api_base, api_type, api_version, max_tokens, temperature, etc.
        Returns:
            the instantiated model
        """
        if model_name in self.models:
            raise ValueError(f"Model {model_name} already registered")

        if model_type is None:
            model_type = model_name

        if model_type in GPT.AVAILABLE_MODELS:
            llm = GPT(model_type, **generation_options)
        else:
            llm = VLLM(model_type, **generation_options)

        self.models[model_name] = llm
        return llm

    @property
    def total_cost(self):
        return sum([llm.total_cost for llm in self.models.values()])

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            config = _replace_env_vars(yaml.safe_load(f))
        return cls(config=config)

    def _load_from_configs(self, configs: List[Dict]):
        for config in configs:
            name = config.pop("name")  # how you refer to the model
            model = config.pop("model", name)  # the api model name
            self.register(name, model, **config)

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, model_name):
        return self.models[model_name]

    def __contains__(self, model_name):
        return model_name in self.models

    def get(self, model_name, default=None):
        if model_name in self:
            return self[model_name]
        return default


@contextmanager
def isolated_cost(llms: Union[LLMRegistry, LLM, List[LLM]], add_cost_to_total: bool = False):
    if isinstance(llms, LLM):
        llms = [llms]
    elif isinstance(llms, LLMRegistry):
        llms = list(llms.models.values())

    previous_costs = {llm: llm.total_cost for llm in llms}
    try:
        for llm in llms:
            llm.total_cost = 0.0
        yield
    finally:
        for llm in llms:
            if add_cost_to_total:
                llm.total_cost += previous_costs[llm]
            else:
                llm.total_cost = previous_costs[llm]


def _replace_env_vars(data):
    pattern = re.compile(r'\$\{(.*)\}')
    if isinstance(data, dict):
        for key in data:
            data[key] = _replace_env_vars(data[key])
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = _replace_env_vars(data[i])
    elif isinstance(data, str):
        match = pattern.search(data)
        if match:
            var = match.group(1)
            data = data.replace('${' + var + '}', os.getenv(var))
    return data