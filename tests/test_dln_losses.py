import numpy as np
import pytest
from dln.loss import LLoss, LossRegistry, NumberPresenceLoss, ExactMatchLoss


def test_available_losses():
    assert "number_presence_loss" in LossRegistry.available_losses()
    for loss in LossRegistry.available_losses():
        assert isinstance(LossRegistry.instantiate(loss), LLoss)


def test_instantiate_loss_from_str():
    loss = LossRegistry.instantiate("exact_match_loss")
    assert isinstance(loss, ExactMatchLoss)
    with pytest.raises(ValueError, match="Unknown loss type: UnknownLoss"):
        LossRegistry.instantiate("UnknownLoss")


def test_exact_match_loss():
    y = ["a", "b", "c", "a", "b", "c"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    exact_match_loss = ExactMatchLoss(lambda x: x)
    losses = exact_match_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_exact_match_loss_no_postproc():
    y = ["A", "B", "C", "a", "b", "c"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    exact_match_loss = ExactMatchLoss()
    losses = exact_match_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_exact_match_loss_postproc():
    y = ["A", "B", "C", "A", "B", "C"]
    y_hat = ["a", "a", "a", "b", "b", "c"]
    exact_match_loss = ExactMatchLoss(lambda x: x.lower())
    losses = exact_match_loss(y, y_hat)
    np.testing.assert_array_equal(losses, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert y == ["A", "B", "C", "A", "B", "C"]  # no side effect


def test_exact_match_loss_postproc_property():
    exact_match_loss = ExactMatchLoss(lambda x: x.upper())
    assert exact_match_loss.postproc("abc") == "ABC"

    exact_match_loss = ExactMatchLoss()
    assert exact_match_loss.postproc("abc") == "abc"


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
    assert not number_presence_loss("1234.0", 1234)
    assert not number_presence_loss("1234", 1234.0)
    assert not number_presence_loss("01234,000", 1234000)
    assert not number_presence_loss("Answer 1234 in it.", "1234")
    assert not number_presence_loss("Answer\n1234", 1234)
    assert not number_presence_loss("Answer 01234 in it.", 1234)
    assert not number_presence_loss("Answer 01234,000 in it.", 1234000)
    assert not number_presence_loss("Answer 01234.0 in it.", 1234)
    assert not number_presence_loss("Answer 01234 in it.", 1234.0)
    assert not number_presence_loss("Answer 01234 in it.", 1234.0)
    assert not number_presence_loss("Answer=01234.", 1234)
    assert not number_presence_loss("$1234.00", 1234)
    assert not number_presence_loss("$1234.50", 1234.5)
    assert number_presence_loss("$1234.50", 1234)
    assert number_presence_loss("$12.34", 1234)
    assert number_presence_loss("12340", "1234")
    assert number_presence_loss("Answer 12340 not in it.", 1234)
    assert number_presence_loss("Answer test_1234 not in it.", "1234")
    assert number_presence_loss("Answer 101234 not in it.", 1234)
    assert number_presence_loss("Answer 0123.4 not in it.", 1234)
