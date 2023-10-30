from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from dln.operator import GPT, VLLM, LLMRegistry, isolated_cost


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
    llm = LLMRegistry.instantiate_llm("text-davinci-003", **gpt_api_config)
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
