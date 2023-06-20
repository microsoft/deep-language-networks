import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from dln.vi.sampler import PosteriorSampler, PromptSampler


def test_sample_q_p(backward_info):
    inputs, y, y_hat, losses = backward_info
    sampler = PromptSampler()
    mock_eval_fn = MagicMock(return_value=["new prompt 1", "new prompt 2"])
    sampler.evaluate_func = mock_eval_fn
    prompt = "test prompt"
    num_samples = 2
    held_out_half = False
    prompts = sampler.sample_q_p(
        inputs, y, y_hat, losses, prompt, num_samples, held_out_half
    )

    q_prompt = mock_eval_fn.call_args[0][0][
        0
    ]  # rendered template sent to evaluate_func

    success_block = re.findall(r"# Student successes(.*?)\n\n", q_prompt, re.DOTALL)[0]
    assert "test_1" in success_block
    assert "test_3" in success_block
    assert "test_2" not in success_block
    assert "test_4" not in success_block

    error_block = re.findall(r"# Student errors(.*?)\n\n", q_prompt, re.DOTALL)[0]
    assert "test_2" in error_block
    assert "test_4" in error_block
    assert "test_1" not in error_block
    assert "test_3" not in error_block

    np.testing.assert_array_equal(
        prompts, ["test prompt", "new prompt 1", "new prompt 2"]
    )


def test_sample_q_p_hold_out_half(backward_info):
    inputs, y, y_hat, losses = backward_info
    sampler = PromptSampler()
    mock_eval_fn = MagicMock(return_value=["new prompt 1", "new prompt 2"])
    sampler.evaluate_func = mock_eval_fn
    prompt = "test prompt"
    num_samples = 2
    held_out_half = True
    prompts = sampler.sample_q_p(
        inputs, y, y_hat, losses, prompt, num_samples, held_out_half
    )

    q_prompt = mock_eval_fn.call_args[0][0][
        0
    ]  # rendered template sent to evaluate_func

    success_block = re.findall(r"# Student successes(.*?)\n\n", q_prompt, re.DOTALL)[0]
    error_block = re.findall(r"# Student errors(.*?)\n\n", q_prompt, re.DOTALL)[0]

    success_examples = [i for i in y if i in success_block]
    error_examples = [i for i in y_hat if i in error_block]

    assert len(success_examples + error_examples) == 2
    assert "test_2" not in success_block
    assert "test_4" not in success_block
    assert "test_1" not in error_block
    assert "test_3" not in error_block
    np.testing.assert_array_equal(
        prompts, ["test prompt", "new prompt 1", "new prompt 2"]
    )


def test_sample_q_h(backward_info):
    inputs, y, _, _ = backward_info
    h = ["test 1", "test2", "test 3", "test4"]
    num_samples = 2
    sampler = PosteriorSampler("suffix_forward_tbs")
    mock_eval_fn = MagicMock(
        # h * num_samples
        return_value=[
            "test 1.1",
            "test 1.2",
            "test 2.1",
            "test 2.2",
            "test 3.1",
            "test 3.2",
            "test 4.1",
            "test 4.2",
        ]
    )
    sampler.evaluate_func = mock_eval_fn
    prompt = "test prompt"
    next_prompt = "test next prompt"
    h_hat = sampler.sample_q_h(
        inputs,
        y,
        h,
        prompt,
        next_prompt,
        num_samples,
    )
    np.testing.assert_equal(
        h_hat,
        [
            ["test 1.1", "test 1.2"],
            ["test 2.1", "test 2.2"],
            ["test 3.1", "test 3.2"],
            ["test 4.1", "test 4.2"],
        ],
    )
