import numpy as np

from dln.operator import instantiate_tokenizer
from dln.score import LogProbsScore, OutputClasses


def test_logprobs_score_with_output_classes(score_requests, top_logprobs, mock_llm):
    mock_llm.generate = top_logprobs
    tokenizer = instantiate_tokenizer("text-davinci-003")
    logprobs_score = LogProbsScore(tokenizer, mock_llm)

    logprobs = logprobs_score.score_requests(
        score_requests, output_classes=OutputClasses(protos=["a|A", "b|B"])
    )

    np.testing.assert_almost_equal(logprobs.targets, [-8.6746863, -0.4428973])
    np.testing.assert_almost_equal(
        logprobs.contexts,
        [
            [9.99829143e-01, 1.70856546e-04],
            [6.42173164e-01, 3.57826836e-01],
        ],
    )


def test_logprobs_score_without_output_classes(score_requests, raw_logprobs, mock_llm):
    mock_llm.generate = raw_logprobs
    tokenizer = instantiate_tokenizer("text-davinci-003")
    logprobs_score = LogProbsScore(tokenizer, mock_llm)
    
    logprobs = logprobs_score.score_requests(score_requests)

    np.testing.assert_almost_equal(logprobs.targets, [-0.7682657, -0.7632834])
    np.testing.assert_almost_equal(logprobs.contexts, [-2.8217665, -2.73069])
