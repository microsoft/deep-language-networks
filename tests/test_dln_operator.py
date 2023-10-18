from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from dln.operator import GPT, VLLM, Connections


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


def test_load_connections():
    gpt_api_config = {
        "api_key": "gpt3-key",
        "api_base": "https://gpt-3-api.com",
        "api_type": "azure",
        "api_version": "2023-03-15-preview",
    }
    llama_api_config = {
        "api_key": "llama-key",
        "api_base": "https://llama-api.com",
        "api_type": None,
        "api_version": None,
    }
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
        connections = Connections(config=config)

    assert len(connections) == 2
    gpt = connections.get("gpt-3")
    llama = connections.get("llama2")
    assert isinstance(gpt, GPT)
    assert isinstance(llama, VLLM)
    assert gpt.engine == "text-davinci-003"
    assert llama.engine == "llama2"
    assert gpt.generation_options == gpt_api_config
    assert llama.generation_options == llama_api_config


def test_unknow_connections():
    config = [
        {
            "name": "gpt-3",
            "model": "text-davinci-003",
            "api_key": "gpt3-key",
            "api_base": "https://gpt-3-api.com",
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
        }
    ]
    connections = Connections(config=config)
    with pytest.raises(KeyError):
        connections["llama2"]
    assert connections.get("llama2") is None
    assert connections.get("llama2", default="default") == "default"


def test_load_connections_from_yaml(tmp_path):
    connections_yaml_content = """
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

    connections_yaml_path = tmp_path / "connections.yaml"
    connections_yaml_path.write_text(connections_yaml_content)

    with patch("dln.operator.instantiate_tokenizer"):
        connections = Connections.from_yaml(connections_yaml_path)

    assert len(connections) == 2

    gpt = connections.get("gpt-3")
    llama = connections.get("llama2")

    assert isinstance(gpt, GPT)
    assert isinstance(llama, VLLM)
    assert gpt.generation_options["api_key"] == "gpt3-key"
    assert llama.generation_options["api_key"] == "llama-key"
