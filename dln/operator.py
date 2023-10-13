from abc import ABC, abstractmethod
from typing import List, Union
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

    def __init__(self, model_name: str, **generation_options):
        self.generation_options = generation_options
        self.engine = model_name

    def __call__(self, inputs: Union[List[str], str], **kwargs) -> List[str]:
        return self.generate(inputs, **kwargs)

    @abstractmethod
    def generate(self, inputs: Union[List[str], str], **kwargs) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, string: str) -> List[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def has_log_probs(self) -> bool:
        raise NotImplementedError
    
    def compute_cost(self, inputs: List[str]) -> float:
        return np.sum(list([len(self.encode(input)) for input in inputs]))


class GPT(LLM):

    CHAT_COMPLETION_MODELS = [
        "gpt-35-turbo",  # azure
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-0613",
    ]

    COMPLETION_MODELS = [
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    ]

    AVAILABLE_MODELS = CHAT_COMPLETION_MODELS + COMPLETION_MODELS

    def __init__(self, model_name: str = "text-davinci-003", **generation_options):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"GPT model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )
        super().__init__(model_name, **generation_options)
        engine_for_encoder = self.engine
        if engine_for_encoder == "gpt-35-turbo":
            engine_for_encoder = "gpt-3.5-turbo"
        self.encoder = instantiate_tokenizer(engine_for_encoder)
        openai.api_version = os.environ.get('OPENAI_API_VERSION')

    def encode(self, string: str) -> List[int]:
        return self.encoder.encode(string)

    @property
    def has_log_probs(self) -> bool:
        return self.engine in self.COMPLETION_MODELS

    @_retry_request(min_wait=4, max_wait=10, max_attempts=100)
    async def _aget_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = await openai.ChatCompletion.acreate(
                    deployment_id=self.engine,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except openai.InvalidRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warn(str(e))
                return str(e)
        else:
            response = await openai.ChatCompletion.acreate(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        if "content" not in response["choices"][0]["message"]:
            return ""

        output = response["choices"][0]["message"]["content"].strip()
        return output

    @_retry_request(min_wait=4, max_wait=10, max_attempts=100)
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
            # Most likely a content filtering error from Azure.
            if "filtering" in str(e):
                logging.warn(str(e))
                # Process each element in the batch individually.
                response = {"choices": []}
                for prompt in prompt_batch:
                    try:
                        response["choices"].append(
                            openai.Completion.create(
                                engine=self.engine,
                                prompt=prompt,
                                logprobs=top_logprobs or 1,
                                **kwargs,
                            )["choices"][0]
                        )
                    except openai.InvalidRequestError as e:
                        response["choices"].append(
                            {
                                "text": str(e),
                                "logprobs": {
                                    "token_logprobs": [0],
                                    "top_logprobs": [{}],
                                    "tokens": {}
                                },
                            }
                        )
            else:
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

    def generate(
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

        if self.engine in self.CHAT_COMPLETION_MODELS:
            if "return_logprobs" in generation_options:
                logging.warn(
                    "return_logprobs is not supported for chat completion models"
                )
                del generation_options["return_logprobs"]

            if async_generation is True:
                # async call api, devide to mini batches to avoid call rate limit
                outputs = []
                for input_batch in self._mini_batch(inputs, batch_size=10):
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

    def generate(
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
    def has_log_probs(self) -> bool:
        return True


def instantiate_model(model_name: str, **generation_options) -> LLM:
    if model_name in GPT.AVAILABLE_MODELS:
        return GPT(model_name, **generation_options)
    return VLLM(model_name, **generation_options)


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
