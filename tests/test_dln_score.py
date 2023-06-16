from unittest.mock import patch

import numpy as np
import tiktoken

from dln.score import LogProbsScore, OutputClasses


def test_logprobs_score_with_output_classes(score_requests, top_logprobs):
    encoder = tiktoken.encoding_for_model("text-davinci-003")
    logprobs_score = LogProbsScore(encoder)

    with patch("dln.score.lops.forward_evaluate", top_logprobs):
        output_logprobs, output_distribs = logprobs_score.score_requests(
            score_requests, output_classes=OutputClasses(protos=["a|A", "b|B"])
        )

    np.testing.assert_almost_equal(output_logprobs, [-8.6746863, -0.4428973])
    np.testing.assert_almost_equal(
        output_distribs,
        [
            [9.99829143e-01, 1.70856546e-04],
            [6.42173164e-01, 3.57826836e-01],
        ],
    )


def test_logprobs_score_without_output_classes(score_requests, raw_logprobs):
    encoder = tiktoken.encoding_for_model("text-davinci-003")
    logprobs_score = LogProbsScore(encoder)
    with patch("dln.score.lops.forward_evaluate", raw_logprobs):
        output_logprobs, output_distribs = logprobs_score.score_requests(score_requests)

    np.testing.assert_almost_equal(output_logprobs, [-0.7682657, -0.7632834])
    np.testing.assert_almost_equal(output_distribs, [-2.8217665, -2.73069])
