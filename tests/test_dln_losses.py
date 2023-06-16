import numpy as np
from dln.loss import ZeroOneLoss


def test_zero_one_loss():
    y = ["a", "b", "c", "a", "b", "c"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    zero_one_loss = ZeroOneLoss(lambda x: x)
    losses = zero_one_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_zero_one_loss_no_postproc():
    y = ["A", "B", "C", "a", "b", "c"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    zero_one_loss = ZeroOneLoss()
    losses = zero_one_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_zero_one_loss_postproc():
    y = ["A", "B", "C", "A", "B", "C"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    zero_one_loss = ZeroOneLoss(lambda x: x.lower())
    losses = zero_one_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert y == ["A", "B", "C", "A", "B", "C"]  # no side effect


def test_zero_one_loss_postproc_property():
    zero_one_loss = ZeroOneLoss(lambda x: x.upper())
    assert zero_one_loss.postproc("abc") == "ABC"

    zero_one_loss = ZeroOneLoss()
    assert zero_one_loss.postproc("abc") == "abc"
