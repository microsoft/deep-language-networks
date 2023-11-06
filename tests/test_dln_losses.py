import numpy as np
import pytest
from dln.loss import LLoss, NumberPresenceLoss, ZeroOneLoss

def test_instantiate_loss_from_str():
    loss = LLoss.instantiate("ZeroOneLoss")
    assert isinstance(loss, ZeroOneLoss)
    with pytest.raises(ValueError, match="Unknown loss type: UnknownLoss"):
        LLoss.instantiate("UnknownLoss")


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


def test_number_presence_loss():
    number_presence_loss = NumberPresenceLoss()
    inputs = [
        "1234",
        "01234",
        "Answer 1234 in it.",
        "Answer\n1234",
        "Answer 01234 in it.",
        "12 34",
        "12340",
        "Answer 12340 not in it.",
        "Answer test_1234 not in it.",
        "Answer 101234 not in it."
    ]
    targets = [1234] * 10
    loss = number_presence_loss(inputs, targets)
    np.testing.assert_array_equal(
        loss, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    )


def test_number_presence_loss_single_value():
    number_presence_loss = NumberPresenceLoss()
    assert not number_presence_loss("1234", 1234)
    assert not number_presence_loss("01234", "1234")
    assert not number_presence_loss("Answer 1234 in it.", "1234")
    assert not number_presence_loss("Answer\n1234", 1234)
    assert not number_presence_loss("Answer 01234 in it.", 1234)
    assert number_presence_loss("12340", "1234")
    assert number_presence_loss("Answer 12340 not in it.", 1234)
    assert number_presence_loss("Answer test_1234 not in it.", "1234")
    assert number_presence_loss("Answer 101234 not in it.", 1234)
