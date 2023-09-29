from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from dln.operator import GPT


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
    mock_api.ChatCompletion.create.return_value = chat_completion_data
    mock_api.Completion.create.return_value = completion_data
    monkeypatch.setattr(openai, "ChatCompletion", mock_api.ChatCompletion)
    monkeypatch.setattr(openai, "Completion", mock_api.Completion)


@pytest.fixture
def mock_openai_api_async(monkeypatch, mock_data):
    chat_completion_data, completion_data = mock_data
    mock_api = MagicMock()
    mock_api.ChatCompletion.acreate = AsyncMock(return_value=chat_completion_data)
    monkeypatch.setattr(openai, "ChatCompletion", mock_api.ChatCompletion)


def test_invalid_model_name():
    with pytest.raises(ValueError):
        GPT("invalid-model-name")


def test_valid_model_name():
    gpt = GPT("text-davinci-003")
    assert gpt.engine == "text-davinci-003"


@pytest.mark.asyncio
async def test_aget_chat_completion_response(mock_openai_api_async):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = await gpt.aget_chat_completion_response(prompt)
    assert "Montreal" in response


def test_get_chat_completion_response(mock_openai_api):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = gpt.get_chat_completion_response(prompt)
    assert "Montreal" in response


def test_get_completion_response(mock_openai_api):
    gpt = GPT("text-davinci-003")
    prompt = "What is the largest city in Quebec?"
    response = gpt.get_completion_response([prompt])
    assert "Montreal" in response[0]
