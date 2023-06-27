from unittest.mock import patch

import numpy as np

from dln.operator import forward_instantiate
from dln.score import OutputClasses
from dln.vi.layers import PriorLayer, ResidualPriorLayer


def test_apply_residual_without_template():
    inputs = np.array(["input1", "input2", "input3"])
    outputs = np.array(["output1", "output2", "output3"])
    residual_prior_layer = ResidualPriorLayer(
        forward_template="suffix_forward",
        init="A task description",
    )
    result = residual_prior_layer.apply_residual(outputs, inputs)
    expected_outputs = np.array(
        [
            "input1\nYour thoughts were:\noutput1",
            "input2\nYour thoughts were:\noutput2",
            "input3\nYour thoughts were:\noutput3",
        ]
    )
    np.testing.assert_equal(result, expected_outputs)


def test_apply_residual_with_template():
    inputs = np.array(["input1", "input2", "input3"])
    outputs = np.array(["output1", "output2", "output3"])
    residual_prior_layer = ResidualPriorLayer(
        forward_template="suffix_forward",
        init="A task description",
    )
    result = residual_prior_layer.apply_residual(outputs, inputs, use_template=True)
    expected_outputs = np.array(
        [
            "input1\n\nA task description\nYour thoughts were:\noutput1",
            "input2\n\nA task description\nYour thoughts were:\noutput2",
            "input3\n\nA task description\nYour thoughts were:\noutput3",
        ]
    )
    np.testing.assert_equal(result, expected_outputs)


def test_log_p_with_output_classes(top_logprobs):
    forward_instantiate()
    inputs = ["1 + 1", "1 * 1"]
    outputs = ["B", "A"]
    output_classes = OutputClasses(protos=["a|A", "b|B"])
    prior_layer = PriorLayer(forward_template="suffix_forward", init="")
    with patch("dln.score.forward_evaluate", top_logprobs):
        logp = prior_layer.log_p(
            inputs, outputs, output_classes=output_classes
        )
    np.testing.assert_almost_equal(logp.targets, [-8.67468626, -0.44289729])
    np.testing.assert_almost_equal(
        logp.contexts,
        [
            [9.99829143e-01, 1.70856546e-04],
            [6.42173164e-01, 3.57826836e-01],
        ],
    )


def test_log_p_without_output_classes(raw_logprobs, score_requests):
    forward_instantiate()
    inputs = [s.context for s in score_requests]
    outputs = ["B", "A"]
    prior_layer = PriorLayer(forward_template="suffix_forward", init="")
    with patch("dln.score.forward_evaluate", raw_logprobs):
        logp = prior_layer.log_p(inputs, outputs)
    np.testing.assert_almost_equal(logp.targets, [-1.48348267, -1.47351816])


def test_forward_with_output_class(top_logprobs):
    forward_instantiate()
    inputs = ["1 + 1", "1 * 1"]
    output_classes = OutputClasses(protos=["A|a", "B|b"])
    prior_layer = PriorLayer(forward_template="suffix_forward", init="")
    with patch("dln.score.forward_evaluate", top_logprobs):
        result = prior_layer.forward(inputs, output_classes)
    np.testing.assert_equal(result, ["A", "A"])


def test_forward_without_output_class(text_outputs):
    forward_instantiate()
    inputs = ["1 + 1", "1 * 1"]
    prior_layer = PriorLayer(forward_template="suffix_forward", init="")
    with patch("dln.vi.layers.forward_evaluate", text_outputs):
        result = prior_layer.forward(inputs)
    np.testing.assert_equal(result, ["A", "A"])


def test_forward_strip_double_newlines():
    forward_instantiate()
    inputs = ["1 + 1"]
    prior_layer = PriorLayer(forward_template="suffix_forward", init="")
    text_output = lambda *args, **kwargs: ["A\n\n"]
    with patch("dln.vi.layers.forward_evaluate", text_output):
        result = prior_layer.forward(inputs, strip_double_newlines=True)
    np.testing.assert_equal(result, ["A\n"])
