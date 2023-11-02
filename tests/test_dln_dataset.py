from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from dln.dataset import Dataset, init_dataset, option_shuffle


def test_init_dataset_subj():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("subj", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is subjective or objective."
    )
    assert dataset.dataset_name == "subj"
    assert dataset.output_classes.protos == ["subjective", "objective"]
    assert dataset.dev_size == 256


def test_init_dataset_mpqa():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("mpqa", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following review, then choose whether it is negative or positive."
    )
    assert dataset.dataset_name == "mpqa"
    assert dataset.output_classes.protos == ["negative", "positive"]
    assert dataset.dev_size == 256


def test_init_dataset_trec():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("trec", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following question, then choose whether it is about a description, entity, expression, human, location or number."
    )
    assert dataset.dataset_name == "trec"
    assert dataset.output_classes.protos == [
        "description",
        "entity",
        "expression",
        "human",
        "location",
        "number",
    ]
    assert dataset.dev_size == 256


def test_init_dataset_disaster():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("disaster", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is relevant to a disaster."
    )
    assert dataset.dataset_name == "disaster"
    assert dataset.output_classes.protos == ["no", "yes"]
    assert dataset.dev_size == 250


def test_init_dataset_airline():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("airline", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is positive, negative, or neutral."
    )
    assert dataset.dataset_name == "airline"
    assert dataset.output_classes.protos == ["positive|Positive", "negative|Negative", "neutral|Neutral"]
    assert dataset.dev_size == 250


def test_init_dataset_hyperbaton():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("hyperbaton", 42, "./data")
    assert dataset.instruction == "Which sentence has the correct adjective order:"
    assert dataset.dataset_name == "hyperbaton"
    assert dataset.output_classes.protos == ["a|A", "b|B"]
    assert dataset.dev_size == 300


def test_init_dataset_navigate():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset("navigate", 42, "./data")
    assert (
        dataset.instruction
        == "Read the following sentence, then determine whether you return to the starting point."
    )
    assert dataset.dataset_name == "navigate"
    assert dataset.output_classes.protos == ["yes|Yes", "no|No"]
    assert dataset.dev_size == 375


def test_init_dataset_date_understanding():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset(
            "date_understanding", 42, "./data"
        )
    assert dataset.instruction == "Infer the date from context."
    assert dataset.dataset_name == "date_understanding"
    assert dataset.output_classes.protos == ["a|A", "b|B", "c|C", "d|D", "e|E", "f|F"]
    assert dataset.dev_size == 119


def test_init_dataset_logical_deduction_seven_objects():
    with patch.object(Dataset, "load_dataset", MagicMock):
        dataset = init_dataset(
            "logical_deduction_seven_objects", 42, "./data"
        )
    assert (
        dataset.instruction
        == "A seven-object logical deduction task which requires deducing the order of a sequence of objects."
    )
    assert dataset.dataset_name == "logical_deduction_seven_objects"
    assert dataset.output_classes.protos == ["a|A", "b|B", "c|C", "d|D", "e|E", "f|F", "g|G"]
    assert dataset.dev_size == 225


def test_init_dataset_not_found():
    with pytest.raises(AssertionError, match=r"Dataset test not found"):
        init_dataset("test", 42, "./data")


def test_option_shuffle():
    data_point = {
        "input": "This is a question.\nOptions:\n(A) Option A\n(B) Option B\n(C) Option C\n(D) Option D",
        "target": "(C)"
    }
    target_input = "Option C"
    rng = np.random.default_rng(42)

    new_data_point = option_shuffle(data_point, rng)

    options = new_data_point['input'].split('\n')[2:]
    target_option = [i for i in options if target_input in i][0]
    assert len(options) == 4
    assert new_data_point["target"] in target_option
