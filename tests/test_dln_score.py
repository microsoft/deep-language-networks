import numpy as np

from dln.score import LogProbsScore, OutputClasses


def test_logprobs_score_with_output_classes(score_requests, top_logprobs, mock_llm_func):
    mock_llm = mock_llm_func("text-davinci-003")
    mock_llm._generate = top_logprobs
    logprobs_score = LogProbsScore(mock_llm)

    logprobs = logprobs_score.score_requests(
        score_requests, output_classes=OutputClasses(protos=["a|A", "b|B"])
    )

    np.testing.assert_almost_equal(logprobs.logp_targets, [-8.6746863, -0.4428973])
    np.testing.assert_almost_equal(
        logprobs.distribution,
        [
            [9.99829143e-01, 1.70856546e-04],
            [6.42173164e-01, 3.57826836e-01],
        ],
    )


def test_logprobs_score_without_output_classes(score_requests, raw_logprobs, mock_llm_func):
    mock_llm = mock_llm_func("text-davinci-003")
    mock_llm._generate = raw_logprobs
    logprobs_score = LogProbsScore(mock_llm)

    logprobs = logprobs_score.score_requests(score_requests)

    np.testing.assert_almost_equal(logprobs.logp_targets, [-0.7682657, -0.7632834])
    np.testing.assert_almost_equal(logprobs.distribution, [-2.8217665, -2.73069])
