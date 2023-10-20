from unittest.mock import patch

import numpy as np
import pytest

from dln.loss import ZeroOneLoss
from dln.postprocessing import postprocess_prediction
from dln.score import LogProbs, OutputClasses
from dln.vi.layers import PriorLayer, ResidualPriorLayer
from dln.vi.model import VILModel


@pytest.fixture
def loss_fn():
    loss_fn = ZeroOneLoss(postproc=postprocess_prediction)
    return loss_fn


@pytest.fixture
def log_likes():
    return np.array(
        [
            [[0.2, 0.3], [0.4, 0.1]],
            [[0.5, 0.5], [0.3, 0.2]],
        ]
    )


@pytest.fixture
def class_weights():
    return np.array([[0.6, 0.4], [0.8, 0.2]])


@pytest.fixture
def log_p_fn():
    def log_p(
        self,
        inputs,
        targets,
        prompts=None,
        output_classes=None,
        agg="max",
    ):
        np.random.seed(42)
        logprobs = LogProbs(
            np.random.rand(len(inputs)),
            np.random.rand(len(inputs), len(output_classes))
            if output_classes else None
        )
        return logprobs

    return log_p


@pytest.fixture
def mock_prompt_sampler():
    class MockPromptSampler:
        def sample_q_p(self, *args, **kwargs):
            return np.array(["prompt 1", "prompt 2"])
    return MockPromptSampler()


@pytest.fixture
def mock_posterior_sampler():
    class MockPosteriorSampler:
        def sample_q_h(self, *args, **kwargs):
            return np.array(
                [
                    ["test 1.1", "test 1.2"],
                    ["test 2.1", "test 2.2"],
                    ["test 3.1", "test 3.2"],
                    ["test 4.1", "test 4.2"],
                ]
            )
    return MockPosteriorSampler()


def test_memory(loss_fn):
    model = VILModel(loss_fn, use_memory=2)
    assert model.get_from_memory(0).size == 0
    assert model.get_from_memory(1).size == 0
    with pytest.raises(BaseException):
        model.get_from_memory(2)
    model.add_to_memory("test1", "test2", 0.5)
    model.add_to_memory("test3", "test4", 0.2)
    model.add_to_memory("test5", "test6", 0.7)
    np.testing.assert_array_equal(model.get_from_memory(0), ["test1", "test5"])


def test_compute_elbo_score(loss_fn, log_likes, class_weights):
    model = VILModel(loss_fn)
    elbo_score = model.compute_elbo_score(log_likes)
    expected_output = [[0.35, 0.4], [0.35, 0.15]]
    assert np.allclose(elbo_score, expected_output)

    elbo_score = model.compute_elbo_score(log_likes, class_weights)
    assert np.allclose(elbo_score, [0.37, 0.33])


def test_sample_hidden_states(loss_fn, mock_prompt_sampler, mock_posterior_sampler):
    np.random.seed(42)
    inputs = np.array(["test-1", "test-2", "test-3", "test-4"])
    y = np.array(["test_1", "test_2", "test_3", "test_4"])
    h = np.array(["test 1", "test2", "test 3", "test4"])
    num_h_samples = 2
    model = VILModel(
        loss_fn,
        prompt_sampler_1=mock_prompt_sampler,
        prompt_sampler_2=mock_prompt_sampler,
        posterior_sampler=mock_posterior_sampler,
        num_h_samples=num_h_samples,
    )
    total_h_samples = len(inputs) * num_h_samples
    mock_l2_log_p = LogProbs(
        np.random.rand(total_h_samples),
        np.random.rand(total_h_samples, num_h_samples),
    )
    mock_l1_log_p = LogProbs(np.random.rand(total_h_samples), None)

    with patch.object(
        PriorLayer, "log_p", return_value=mock_l2_log_p
    ), patch.object(
        ResidualPriorLayer, "log_p", return_value=mock_l1_log_p
    ):
        hidden_states = model.sample_hidden_states(x=inputs, y=y, h1=h)

    residual_h_tilde_1, h_tilde_1, h_tilde_1_star, weights = hidden_states

    np.testing.assert_equal(
        residual_h_tilde_1,
        [
            [
                "test-1\nYour thoughts were:\ntest 1.1",
                "test-1\nYour thoughts were:\ntest 1.2",
            ],
            [
                "test-2\nYour thoughts were:\ntest 2.1",
                "test-2\nYour thoughts were:\ntest 2.2",
            ],
            [
                "test-3\nYour thoughts were:\ntest 3.1",
                "test-3\nYour thoughts were:\ntest 3.2",
            ],
            [
                "test-4\nYour thoughts were:\ntest 4.1",
                "test-4\nYour thoughts were:\ntest 4.2",
            ],
        ],
    )
    np.testing.assert_equal(h_tilde_1, mock_posterior_sampler.sample_q_h())
    np.testing.assert_equal(
        h_tilde_1_star, ["test 1.2", "test 2.2", "test 3.1", "test 4.2"]
    )
    np.testing.assert_almost_equal(
        weights,
        [
            [0.28796663, 0.71203337],
            [0.45481729, 0.54518271],
            [0.63320434, 0.36679566],
            [0.40828206, 0.59171794],
        ],
    )


def test_inference_one_layer(loss_fn, backward_info, log_p_fn, mock_prompt_sampler, mock_posterior_sampler, mock_logprobs_score):
    np.random.seed(42)
    inputs, y, y_hat, losses = backward_info
    num_h_samples = 2
    num_p_samples = 2
    output_classes = OutputClasses(protos=["A", "B"])
    model = VILModel(
        loss_fn,
        prompt_sampler_1=mock_prompt_sampler,
        prompt_sampler_2=mock_prompt_sampler,
        posterior_sampler=mock_posterior_sampler,
        logprobs_score=mock_logprobs_score,
        output_classes=output_classes,
        num_h_samples=num_h_samples,
        num_p_samples=num_p_samples,
        two_layers=False,
    )
    with patch.object(PriorLayer, "log_p", log_p_fn):
        elbo, _, p2 = model.inference_one_layer(inputs, y, y_hat, losses)
    np.testing.assert_almost_equal(elbo, 0.64288586)
    assert p2 == "prompt 2"


@pytest.mark.parametrize(
    "train_p1, train_p2, expec_best_p1_elbo, expec_best_p2_elbo, expec_best_p1, expec_best_p2",
    [
        (True, False, 0.44168435, 0.0, "prompt 2", ""),  # Train p1
        (False, True, 0.0, 0.54359470, "", "prompt 2"),  # Train p2
        (True, True, 0.44168435, 0.54359470, "prompt 2", "prompt 2"),  # Train e2e
    ],
)
def test_inference_vi(
    loss_fn,
    backward_info,
    log_p_fn,
    mock_prompt_sampler,
    mock_posterior_sampler,
    train_p1,
    train_p2,
    expec_best_p1_elbo,
    expec_best_p2_elbo,
    expec_best_p1,
    expec_best_p2,
):
    inputs, y, y_hat, losses = backward_info
    h1 = np.array(["test 1", "test2", "test 3", "test4"])
    num_h_samples = 2
    num_p_samples = 2
    output_classes = OutputClasses(protos=["A", "B"])
    model = VILModel(
        loss_fn,
        prompt_sampler_1=mock_prompt_sampler,
        prompt_sampler_2=mock_prompt_sampler,
        posterior_sampler=mock_posterior_sampler,
        output_classes=output_classes,
        num_h_samples=num_h_samples,
        num_p_samples=num_p_samples,
        train_p1=train_p1,
        train_p2=train_p2,
    )
    with patch.object(
        PriorLayer, "log_p", log_p_fn
    ), patch.object(
        ResidualPriorLayer, "log_p", log_p_fn
    ):
        r_h1 = model.encoder_l1.apply_residual(h1, inputs)
        best_p1_elbo, best_p2_elbo, best_p1, best_p2 = model.inference_vi(
            inputs, h1, r_h1, y, y_hat, losses
        )

    np.testing.assert_almost_equal(best_p2_elbo, expec_best_p2_elbo)
    np.testing.assert_almost_equal(best_p1_elbo, expec_best_p1_elbo)
    assert best_p1 == expec_best_p1
    assert best_p2 == expec_best_p2


@pytest.mark.parametrize(
    "train_p1, train_p2, expec_elbo, expec_best_p1, expec_best_p2, expec_loss_mean, expec_elbo1, expec_elbo2",
    [
        (True, False, 0.44168435, 'prompt 2', '', 0.5, 0.44168435, 0.0),  # Train p1
        (False, True, 0.54359470, '', 'prompt 2', 0.5, 0.0, 0.54359470),  # Train p2
        (True, True, 0.98527906, 'prompt 2', 'prompt 2', 0.5, 0.44168435, 0.54359470),  # Train e2e
    ],
)
def test_forward_two_layers(
    loss_fn,
    backward_info,
    log_p_fn,
    mock_prompt_sampler,
    mock_posterior_sampler,
    train_p1,
    train_p2,
    expec_elbo,
    expec_best_p1,
    expec_best_p2,
    expec_loss_mean,
    expec_elbo1,
    expec_elbo2,
):
    inputs, y, y_hat, _ = backward_info
    h1 = np.array(["test 1", "test2", "test 3", "test4"])
    num_h_samples = 2
    num_p_samples = 2
    output_classes = OutputClasses(protos=["A", "B"])
    model = VILModel(
        loss_fn,
        prompt_sampler_1=mock_prompt_sampler,
        prompt_sampler_2=mock_prompt_sampler,
        posterior_sampler=mock_posterior_sampler,
        output_classes=output_classes,
        num_h_samples=num_h_samples,
        num_p_samples=num_p_samples,
        train_p1=train_p1,
        train_p2=train_p2,
        two_layers=True,
    )
    with patch.object(
        PriorLayer, "forward", return_value=y_hat
    ), patch.object(
        PriorLayer, "log_p", log_p_fn
    ), patch.object(
        ResidualPriorLayer, "forward", return_value=h1
    ), patch.object(
        ResidualPriorLayer, "log_p", log_p_fn
    ):
        elbo, best_p1, best_p2, loss_mean, elbo1, elbo2 = model.forward(inputs, y)

    assert best_p1 == expec_best_p1
    assert best_p2 == expec_best_p2
    np.testing.assert_almost_equal(elbo, expec_elbo)
    np.testing.assert_almost_equal(elbo1, expec_elbo1)
    np.testing.assert_almost_equal(elbo2, expec_elbo2)
    np.testing.assert_almost_equal(loss_mean, expec_loss_mean)


def test_forward_one_layer(
    loss_fn,
    backward_info,
    log_p_fn,
    mock_prompt_sampler,
    mock_posterior_sampler,
    mock_llm,
):
    inputs, y, y_hat, _ = backward_info
    num_h_samples = 2
    num_p_samples = 2
    output_classes = OutputClasses(protos=["A", "B"])
    model = VILModel(
        loss_fn,
        forward_evaluate=mock_llm,
        prompt_sampler_1=mock_prompt_sampler,
        prompt_sampler_2=mock_prompt_sampler,
        posterior_sampler=mock_posterior_sampler,
        output_classes=output_classes,
        num_h_samples=num_h_samples,
        num_p_samples=num_p_samples,
        train_p1=False,
        train_p2=False,
        two_layers=False,
    )
    with patch.object(
        PriorLayer, "forward", return_value=y_hat
    ), patch.object(
        PriorLayer, "log_p", log_p_fn
    ):
        elbo, best_p1, best_p2, loss_mean, elbo1, elbo2 = model.forward(inputs, y)
    assert best_p1 == None
    assert best_p2 == "prompt 2"
    np.testing.assert_almost_equal(elbo, 0.64288586)
    np.testing.assert_almost_equal(elbo1, 0.0)
    np.testing.assert_almost_equal(elbo2, 0.64288586)
    np.testing.assert_almost_equal(loss_mean, 0.5)


@pytest.mark.parametrize(
    "train_p1, train_p2, two_layers, expec_l1_calls",
    [
        (True, False, True, 1),
        (False, True, True, 1),
        (True, True, True, 1),
        (True, False, False, 0),
        (False, True, False, 0),
        (True, True, False, 0),
    ],
)
def test_forward_inference(
    loss_fn,
    backward_info,
    train_p1,
    train_p2,
    two_layers,
    expec_l1_calls,
    mock_llm
):
    inputs, _, y_hat, _ = backward_info
    h1 = np.array(["test 1", "test2", "test 3", "test4"])
    num_h_samples = 2
    num_p_samples = 2
    output_classes = OutputClasses(protos=["A", "B"])
    model = VILModel(
        loss_fn,
        forward_evaluate=mock_llm,
        output_classes=output_classes,
        num_h_samples=num_h_samples,
        num_p_samples=num_p_samples,
        train_p1=train_p1,
        train_p2=train_p2,
        two_layers=two_layers,
    )
    # should only require forward pass
    with patch.object(
        PriorLayer, "forward", return_value=y_hat
    ) as l2, patch.object(
        ResidualPriorLayer, "forward", return_value=h1
    ) as l1:
        result = model.forward(inputs)
    np.testing.assert_equal(result, ['test_', 'test', 'test_', 'test'])
    assert l1.call_count == expec_l1_calls
    assert l2.call_count == 1


def test_strip_options(loss_fn):
    model = VILModel(loss_fn)
    input_data = np.array(
        [
            "This is a test\nOptions:\n(A)\n(B)",
            "No options here",
            "Another testOptions:(A)(B)",
        ]
    )
    expected_output = np.array(["This is a test", "No options here", "Another test"])
    output_data = model.strip_options(input_data)
    assert np.array_equal(output_data, expected_output)


def test_strip_prefix(loss_fn):
    model = VILModel(
        loss_fn, strip_prefix_for_hidden="PREFIX:"
    )
    input_data = np.array(
        ["PREFIX: This is a test", "No prefix here", "PREFIX: Another test"]
    )
    expected_output = np.array(["This is a test", "No prefix here", "Another test"])
    output_data = model.strip_prefix(input_data)
    assert np.array_equal(output_data, expected_output)
