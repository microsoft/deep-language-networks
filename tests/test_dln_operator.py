import os
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from dln.operator import GPT, LLM, VLLM, LLMRegistry, _replace_env_vars, isolated_cost


@pytest.fixture
def mock_data():
    chat_completion_data = {"choices": [{"message": {"content": "Montreal"}}]}
    completion_data = {
        "choices": [
            {
                "text": "Montreal",
                "logprobs": {"token_logprobs": [0], "top_logprobs": [{}], "tokens": {}},
            }
        ]
    }
    return chat_completion_data, completion_data


@pytest.fixture
def mock_openai_api(monkeypatch, mock_data):
    chat_completion_data, completion_data = mock_data
    mock_api = MagicMock()
    mock_api.ChatCompletion.acreate = AsyncMock(return_value=chat_completion_data)
    mock_api.Completion.create.return_value = completion_data
    monkeypatch.setattr(openai, "ChatCompletion", mock_api.ChatCompletion)
    monkeypatch.setattr(openai, "Completion", mock_api.Completion)


def test_invalid_model_name():
    with pytest.raises(ValueError):
        GPT("invalid-model-name")


def test_valid_model_name():
    gpt = GPT("text-davinci-003")
    assert gpt.engine == "text-davinci-003"


@pytest.mark.asyncio
async def test_aget_chat_completion_response(mock_openai_api):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = await gpt._aget_chat_completion_response(prompt)
    assert "Montreal" in response


def test_get_completion_response(mock_openai_api):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = gpt._get_completion_response([prompt])
    assert "Montreal" in response[0]


@pytest.mark.parametrize("async_generation", [True, False])
def test_generate(mock_openai_api, async_generation):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = gpt.generate(
        inputs=[prompt, prompt],
        batch_size=1,
        async_generation=async_generation,
    )
    assert response == ["Montreal", "Montreal"]


def test_generate_seeds():

    class MockGPT(LLM):
        def generate(self, inputs, **kwargs):
            return kwargs
        def encode(self, string):
            return string
        def has_logprobs(self):
            return True

    prompt = "Prompt test"

    # generate seed from random state
    mock_gpt = MockGPT("mock_model", seed=42)
    kwargs = mock_gpt(inputs=[prompt, prompt])
    assert kwargs["seed"] == 7270  # randomly generated seed 1
    kwargs = mock_gpt(inputs=[prompt, prompt])
    assert kwargs["seed"] == 860  # randomly generated seed 2

    # override seed with provided value
    kwargs = mock_gpt(inputs=[prompt, prompt], seed=-10)
    assert kwargs["seed"] == -10

    # no seed at init, no seed provided
    mock_gpt_no_seed = MockGPT("mock_model")
    kwargs = mock_gpt_no_seed(inputs=[prompt, prompt])
    assert "seed" not in kwargs

    # no seed at init, seed provided
    kwargs = mock_gpt_no_seed(inputs=[prompt, prompt], seed=-20)
    assert kwargs["seed"] == -20


@pytest.mark.parametrize("model_name", [
    "gpt-35-turbo",
    "gpt-3.5-turbo",
    "gpt-35-turbo-instruct",
    "gpt-3.5-turbo-instruct",
])
def test_gpt_35_name_variations_load_tokenizer(model_name):
    gpt = GPT(model_name)
    assert gpt.engine == model_name
    assert gpt.encoder.name == "cl100k_base"


def test_openai_invalid_request_error(monkeypatch):
    mock_api = MagicMock()
    mock_api.Completion.create.side_effect = openai.InvalidRequestError(
        "Invalid request", "param"
    )
    monkeypatch.setattr(openai, "Completion", mock_api.Completion)
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    with pytest.raises(openai.InvalidRequestError, match="Invalid request"):
        gpt.generate(prompt)


@pytest.fixture
def gpt_api_config():
    return {
        "api_key": "gpt3-key",
        "api_base": "https://gpt-3-api.com",
        "api_type": "azure",
        "api_version": "2023-03-15-preview",
    }


@pytest.fixture
def llama_api_config():
    return {
        "api_key": "llama-key",
        "api_base": "https://llama-api.com",
        "api_type": None,
        "api_version": None,
    }


def test_registry_llm(gpt_api_config):
    from dln.operator import LLMRegistry
    llm_registry = LLMRegistry()
    llm = llm_registry.register("gpt_3", "text-davinci-003", **gpt_api_config)
    assert isinstance(llm, GPT)
    assert llm_registry["gpt_3"] == llm
    assert llm.engine == "text-davinci-003"
    assert llm.generation_options == gpt_api_config
    with patch("dln.operator.instantiate_tokenizer"):
        another_llm = llm_registry.register("llama2", **gpt_api_config)
    assert isinstance(another_llm, VLLM)
    assert llm_registry["llama2"] == another_llm
    assert another_llm.engine == "llama2"


def test_registry_llm_duplicated_name(gpt_api_config):
    registry = LLMRegistry()
    registry.register("text-davinci-003", **gpt_api_config)
    with pytest.raises(
        ValueError,
        match="Model text-davinci-003 already registered"
    ):
        registry.register("text-davinci-003", **gpt_api_config)


def test_load_llms_from_config(gpt_api_config, llama_api_config):
    config = [
        {
            "name": "gpt-3",
            "model": "text-davinci-003",
            **gpt_api_config,
        },
        {
            "name": "llama2",
            # model is not required, use name as default
            **llama_api_config,
        },
    ]

    with patch("dln.operator.instantiate_tokenizer"):
        llm_registry = LLMRegistry(config=config)

    assert len(llm_registry) == 2
    gpt = llm_registry.get("gpt-3")
    llama = llm_registry.get("llama2")
    assert isinstance(gpt, GPT)
    assert isinstance(llama, VLLM)
    assert gpt.engine == "text-davinci-003"
    assert llama.engine == "llama2"
    assert gpt.generation_options == gpt_api_config
    assert llama.generation_options == llama_api_config


def test_get_llm(gpt_api_config):
    config = [
        {
            "name": "gpt-3",
            "model": "text-davinci-003",
            **gpt_api_config,
        }
    ]
    llm_registry = LLMRegistry(config=config)
    assert isinstance(llm_registry["gpt-3"], GPT)
    with pytest.raises(KeyError):
        llm_registry["llama2"]
    assert llm_registry.get("llama2") is None
    assert llm_registry.get("llama2", default="default") == "default"


def test_load_llms_from_yaml(tmp_path):
    llms_yaml_content = """
    - name: gpt-3
      model: text-davinci-003
      api_key: gpt3-key
      api_base: https://gpt-3-api.com
      api_type: azure
      api_version: '2023-03-15-preview'
    - name: llama2
      api_key: llama-key
      api_base: https://llama-api.com
      api_type: null
      api_version: null
    """

    llms_yaml_path = tmp_path / "llms.yaml"
    llms_yaml_path.write_text(llms_yaml_content)

    with patch("dln.operator.instantiate_tokenizer"):
        llm_registry = LLMRegistry.from_yaml(llms_yaml_path)

    assert len(llm_registry) == 2

    gpt = llm_registry.get("gpt-3")
    llama = llm_registry.get("llama2")

    assert isinstance(gpt, GPT)
    assert isinstance(llama, VLLM)
    assert gpt.generation_options["api_key"] == "gpt3-key"
    assert llama.generation_options["api_key"] == "llama-key"


@patch.dict(os.environ, {"TEST": "123", "FOO": "BAR"})
def test_load_llms_from_yaml(tmp_path):
    llms_yaml_content = """
    - name: gpt-3
      model: text-davinci-003
      api_key: gpt3-key
      api_base: ${TEST}
      api_type: TEST
      api_version: ${FOO}
    - name: llama2
      api_key: llama-key
      api_base: https://llama-api.com
      api_type: null
      api_version: null
    """

    llms_yaml_path = tmp_path / "llms.yaml"
    llms_yaml_path.write_text(llms_yaml_content)

    with patch("dln.operator.instantiate_tokenizer"):
        llm_registry = LLMRegistry.from_yaml(llms_yaml_path)

    assert len(llm_registry) == 2

    gpt = llm_registry.get("gpt-3")
    llama = llm_registry.get("llama2")

    assert isinstance(gpt, GPT)
    assert isinstance(llama, VLLM)
    assert gpt.generation_options["api_base"] == "123"
    assert gpt.generation_options["api_type"] == "TEST"
    assert gpt.generation_options["api_version"] == "BAR"
    assert llama.generation_options["api_key"] == "llama-key"


def test_total_cost(gpt_api_config, llama_api_config):
    config = [
        {
            "name": "gpt-3",
            "model": "text-davinci-003",
            **gpt_api_config,
        },
        {
            "name": "llama2",
            # model is not required, use name as default
            **llama_api_config,
        },
    ]

    with patch("dln.operator.instantiate_tokenizer"):
        llm_registry = LLMRegistry(config=config)

    assert llm_registry.total_cost == 0.0
    llm_registry["gpt-3"].total_cost = 2.0
    assert llm_registry.total_cost == 2.0
    llm_registry["llama2"].total_cost = 4.0
    assert llm_registry.total_cost == 6.0


def test_compute_cost_manager(gpt_api_config, mock_openai_api):
    llm = LLMRegistry().register("text-davinci-003", **gpt_api_config)
    assert llm.total_cost == 0.0
    with isolated_cost(llm):  # add_cost_to_total=False by default
        prompt = "What is the largest city in Quebec?"
        response = llm(prompt)
        assert response == ["Montreal"]
        assert llm.total_cost == 37.0
    assert llm.total_cost == 0.0

    with isolated_cost(llm, add_cost_to_total=True):
        prompt = "What is the largest city in Quebec?"
        response = llm(prompt)
        assert response == ["Montreal"]
        assert llm.total_cost == 37.0
    assert llm.total_cost == 37.0


def test_compute_cost_manager_many_llms(gpt_api_config, mock_openai_api):
    registry = LLMRegistry()
    gpt2 = registry.register("text-davinci-002", **gpt_api_config)
    gpt3 = registry.register("text-davinci-003", **gpt_api_config)
    assert gpt2.total_cost == 0.0
    assert gpt3.total_cost == 0.0
    with isolated_cost([gpt2, gpt3]):
        response = gpt2("What is the largest city in Quebec?")
        assert response == ["Montreal"]
        response = gpt3("What is the second-largest city in Canada?")
        assert gpt2.total_cost == 37.0
        assert gpt3.total_cost == 44.0
    assert gpt2.total_cost == 0.0
    assert gpt3.total_cost == 0.0
    with isolated_cost([gpt2, gpt3], add_cost_to_total=True):
        response = gpt2("What is the largest city in Quebec?")
        assert response == ["Montreal"]
        response = gpt3("What is the second-largest city in Canada?")
        assert gpt2.total_cost == 37.0
        assert gpt3.total_cost == 44.0
    assert gpt2.total_cost == 37.0
    assert gpt3.total_cost == 44.0


def test_compute_cost_manager_registry(gpt_api_config, mock_openai_api):
    registry = LLMRegistry()
    gpt2 = registry.register("text-davinci-002", **gpt_api_config)
    gpt3 = registry.register("text-davinci-003", **gpt_api_config)
    assert gpt2.total_cost == 0.0
    assert gpt3.total_cost == 0.0
    with isolated_cost(registry):
        response = gpt2("What is the largest city in Quebec?")
        assert response == ["Montreal"]
        response = gpt3("What is the second-largest city in Canada?")
        assert gpt2.total_cost == 37.0
        assert gpt3.total_cost == 44.0
    assert gpt2.total_cost == 0.0
    assert gpt3.total_cost == 0.0
    with isolated_cost(registry, add_cost_to_total=True):
        response = gpt2("What is the largest city in Quebec?")
        assert response == ["Montreal"]
        response = gpt3("What is the second-largest city in Canada?")
        assert gpt2.total_cost == 37.0
        assert gpt3.total_cost == 44.0
    assert gpt2.total_cost == 37.0
    assert gpt3.total_cost == 44.0


@patch.dict(os.environ, {"TEST": "123"})
def test_replace_env_vars_str():
    assert _replace_env_vars("${TEST}") == "123"


@patch.dict(os.environ, {"TEST": "123"})
def test_replace_env_vars_list():
    assert _replace_env_vars(["${TEST}", "${TEST}"]) == ["123", "123"]


@patch.dict(os.environ, {"TEST": "123"})
def test_replace_env_vars_dict():
    assert _replace_env_vars(
        {"key1": "${TEST}", "key2": "${TEST}"}
    ) == {"key1": "123", "key2": "123"}


@patch.dict(os.environ, {"TEST": "123"})
def test_replace_env_vars_nested():
    assert _replace_env_vars(
        {"key1": ["${TEST}", "${TEST}"], "key2": "${TEST}"}
    ) == {"key1": ["123", "123"], "key2": "123"}


def test_replace_env_vars_no_env():
    assert _replace_env_vars("No env var here") == "No env var here"


def test_replace_env_vars_empty_string():
    assert _replace_env_vars("") == ""
