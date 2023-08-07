# global interpreter
from typing import List
import asyncio
import numpy as np
import openai
import logging
import tiktoken
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

forward_interpreter = None
backward_interpreter = None


class GPT:
    AVAILABLE_MODELS = [
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-0613",
        "any",
    ]

    def __init__(self, model_name="text-davinci-003", **generation_options):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )

        self.generation_options = generation_options
        self.engine = model_name

        if self.engine == "any":
            openai.api_base = "http://0.0.0.0:8082"
            openai.api_key = "any"
            openai.api_type = "openai"
            self.encoder = tiktoken.encoding_for_model("text-davinci-003")
        else:
            self.encoder = tiktoken.encoding_for_model(self.engine)
        openai.api_version = os.environ.get('OPENAI_API_VERSION')

    def encode(self, string):
        return self.encoder.encode(string)

    @property
    def has_log_probs(self):
        return not (
            "gpt-3.5" in self.engine or
            "gpt-4" in self.engine or
            "gpt-35" in self.engine or
            "gpt-4-0613" in self.engine
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    async def aget_chat_completion_response(self, prompt, **kwargs):
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

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    def get_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = openai.ChatCompletion.create(
                    deployment_id=self.engine,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except openai.InvalidRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warn(str(e))
                return str(e)
        else:
            response = openai.ChatCompletion.create(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        if "content" not in response["choices"][0]["message"]:
            return ""

        output = response["choices"][0]["message"]["content"].strip()
        return output

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    def get_completion_response(
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
                                "logprobs": {"token_logprobs": [0], "top_logprobs": [{}], "tokens": {}},
                            }
                        )
            else:
                raise e

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

    async def gather_chat_response(self, inputs, **generation_options):
        outputs = await asyncio.gather(
            *[
                self.aget_chat_completion_response(_input, **generation_options)
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

    def generate(self, inputs, async_generation=True, batch_size=20, **kwargs):
        if type(inputs) is not list:
            inputs = [inputs]

        kwargs.pop("output_space", None)
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)

        if self.engine in ("gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4-0613", "any"):
            if "return_logprobs" in generation_options:
                del generation_options["return_logprobs"]

            if async_generation is True:
                # async call api, devide to mini batches to avoid call rate limit
                outputs = []
                for input_batch in self._mini_batch(inputs, batch_size=10):
                    outputs_batch = asyncio.run(
                        self.gather_chat_response(input_batch, **generation_options)
                    )
                    outputs = outputs + outputs_batch
            else:
                # call api one by one
                outputs = [
                    self.get_chat_completion_response(_input, **generation_options)
                    for _input in inputs
                ]
        else:
            # devide to mini batches (max batch size = 20 according to openai)
            outputs = []
            for input_batch in self._mini_batch(inputs, batch_size=batch_size):
                outputs_batch = self.get_completion_response(
                    input_batch, **generation_options
                )
                outputs = outputs + outputs_batch
        return outputs


def forward_instantiate(model_name="text-davinci-003", **generation_options):
    global forward_interpreter

    if forward_interpreter is None:
        forward_interpreter = GPT(model_name, **generation_options)
    else:
        print("Forward interpreter already instantiated.")
        pass


def backward_instantiate(model_name="text-davinci-003", **generation_options):
    global backward_interpreter

    if backward_interpreter is None:
        backward_interpreter = GPT(model_name, **generation_options)
    else:
        print("Backward interpreter already instantiated.")
        pass


def forward_evaluate(input: List[str], **kwargs):
    return forward_interpreter.generate(input, **kwargs)


def backward_evaluate(input: List[str], **kwargs):
    return backward_interpreter.generate(input, **kwargs)
