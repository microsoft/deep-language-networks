import numpy as np
import pytest
from dln.operator import LLM, instantiate_tokenizer

from dln.score import LogProbsScore, ScoreRequest


@pytest.fixture(autouse=True)
def unset_env_vars(monkeypatch):
    """Unset all environment variables that could be set by the user."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_TYPE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)  # deprecated in favor of OPENAI_BASE_URL
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)


@pytest.fixture
def mock_llm_func():

    def instantiate_llm(model_name=None):
        class MockEncoder:
            def encode(self, string):
                return string

        class MockLLM(LLM):

            def __init__(self, model_name):
                if model_name is None:
                    model_name = "MockLLM"
                    self.encoder = MockEncoder()
                else:
                    self.encoder = instantiate_tokenizer(model_name)
                super().__init__(model_name)

            def _generate(self, inputs, **kwargs):
                return inputs

            def encode(self, string):
                return self.encoder.encode(string)

            def clean_text(self, string):
                return string

            @property
            def has_logprobs(self):
                return True

        return MockLLM(model_name)

    return instantiate_llm


@pytest.fixture
def mock_llm(mock_llm_func):
    return mock_llm_func()


@pytest.fixture
def mock_logprobs_score(mock_llm_func):
    llm = mock_llm_func("text-davinci-003")
    logprobs_score = LogProbsScore(llm)
    return logprobs_score


@pytest.fixture
def backward_info():
    inputs = np.array(["test-1", "test-2", "test-3", "test-4"])
    y = np.array(["test_1", "test_2", "test_3", "test_4"])
    y_hat = np.array(["test_1", "test2", "test_3", "test4"])
    losses = np.array([0.0, 1.0, 0.0, 1.0])
    return inputs, y, y_hat, losses


@pytest.fixture
def score_requests():
    return [
        ScoreRequest(
            context="1 + 1 is:\n(A) 1\n(B) 2\n\nAnswer:",
            target="B",
            payload="B",
        ),
        ScoreRequest(
            context="1 * 1 is:\n(A) 1\n(B) 2\n\nAnswer:",
            target="A",
            payload="A",
        ),
    ]


@pytest.fixture
def text_outputs():
    def logprobs_fn(contexts, *args, **kwargs):
        #  return logprobs in the same order it was requested (contexts)
        logprobs = {
            "1 + 1": "A",
            "1 * 1": "A",
        }
        return [logprobs[context[:5]] for context in contexts]

    return logprobs_fn


@pytest.fixture
def top_logprobs():
    def logprobs_fn(contexts, *args, **kwargs):
        #  return logprobs in the same order it was requested (contexts)
        logprobs = {
            "1 + 1": {
                "Option": -8.876863,
                "Result": -17.299635,
                "choice": -17.710045,
                "<": -17.075796,
                "=": -15.760291,
                "correct": -13.988989,
                " A": -10.678262,
                "A": -3.663905,
                "All": -16.454699,
                "B": -12.343077,
            },
            "1 * 1": {
                "Option": -8.315238,
                "=": -16.698154,
                "A": -11.863415,
                "B": -12.451943,
                "Answer": -7.4255853,
                "answer": -14.647212,
                "Correct": -8.74908,
                "Choice": -13.000805,
                "Yes": -14.741361,
                "b": -17.22967,
            },
        }
        ordered_log_p = [logprobs[context[:5]] for context in contexts]
        return [
            ["0", [ordered_log_p[0]], 2],
            ["0", [ordered_log_p[1]], 2],
        ]

    return logprobs_fn


@pytest.fixture
def raw_logprobs():
    def logprobs_fn(contexts, *args, **kwargs):
        #  return logprobs in the same order it was requested (contexts)
        logprobs = {
            "1 + 1": [
                "1 + 1 is:\n(A) 1\n(B) 2\n\nAnswer:\nB",
                [
                    None,
                    -5.550775,
                    -3.194002,
                    -8.062983,
                    -1.9706848,
                    -0.9759903,
                    -11.239477,
                    -2.745899,
                    -0.030587194,
                    -1.4996661,
                    -0.068833716,
                    -0.009404114,
                    -0.0001532674,
                    -6.5041706e-05,
                    -0.056048736,
                    -0.05334273,
                    -8.41094,
                    -6.9211907,
                    -0.001781753,
                    -0.053041545,
                    -1.4834975,
                ],
                [
                    "1",
                    " +",
                    " 1",
                    " is",
                    ":",
                    "\\n",
                    "(",
                    "A",
                    ")",
                    " 1",
                    "\\n",
                    "(",
                    "B",
                    ")",
                    " 2",
                    "\\n",
                    "\\n",
                    "Answer",
                    ":",
                    "\\n",
                    "B",
                ],
            ],
            "1 * 1": [
                "1 * 1 is:\n(A) 1\n(B) 2\n\nAnswer: A",
                [
                    None,
                    -6.06174,
                    -4.7931056,
                    -8.253801,
                    -2.3915708,
                    -0.5870681,
                    -10.741921,
                    -3.3388677,
                    -0.011392174,
                    -0.86958236,
                    -0.11698982,
                    -0.48095098,
                    -0.002377014,
                    -8.3404535e-05,
                    -1.417262,
                    -0.027041545,
                    -5.510647,
                    -4.546986,
                    -0.0010610583,
                    -0.053041545,
                    -1.4735329,
                ],
                [
                    "1",
                    " *",
                    " 1",
                    " is",
                    ":",
                    "\\n",
                    "(",
                    "A",
                    ")",
                    " 1",
                    "\\n",
                    "(",
                    "B",
                    ")",
                    " 2",
                    "\\n",
                    "\\n",
                    "Answer",
                    ":",
                    "\\n",
                    "A",
                ],
            ],
        }
        return [logprobs[context[:5]] for context in contexts]

    return logprobs_fn
