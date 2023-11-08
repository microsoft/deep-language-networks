import json
import re
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from dln.dataset import init_dataset, option_shuffle



@pytest.fixture
def mock_bbh_data():
    def build_bbh_data(data_path, dataset_name, bbh_size=10, bb_minus_bbh_size=10):
        bbh_dir = data_path / "bbh"
        bbh_dir.mkdir()
        bb_minus_bbh_dir = data_path / "bb_minus_bbh"
        bb_minus_bbh_dir.mkdir()
        bbh_file = bbh_dir / f"{dataset_name}.json"
        bb_minus_bbh_file = bb_minus_bbh_dir / f"{dataset_name}.json"
        bbh_examples = {
            "examples": [{
                "input": "A sentence:\nOptions:\n(A) option A\n(B) option B",
                "target": "(A)"
            }] * bbh_size
        }
        bb_minus_bbh_examples = {
            "examples": [{
                "input": "Another sentence:\nOptions:\n(A) option A\n(B) option B",
                "target": "(B)"
            }] * bb_minus_bbh_size
        }
        bbh_file.write_text(json.dumps(bbh_examples))
        bb_minus_bbh_file.write_text(json.dumps(bb_minus_bbh_examples))
    return build_bbh_data


@pytest.fixture
def mock_ordered_prompt_data():
    def build_ordered_prompt_data(data_path, dataset_name, data_size=1000):
        ordered_prompt_dir = data_path / "ordered_prompt"
        ordered_prompt_dir.mkdir()
        dataset_dir = ordered_prompt_dir / dataset_name
        dataset_dir.mkdir()
        train = dataset_dir / "train.jsonl"
        dev = dataset_dir / "dev_subsample.jsonl"
        test = dataset_dir / "test.jsonl"
        train.write_text('\n'.join(['{"sentence": "train sentence", "label": "0"}'] * data_size))
        dev.write_text('\n'.join(['{"sentence": "dev sentence", "label": "1"}'] * data_size))
        test.write_text('\n'.join(['{"sentence": "test sentence", "label": "2"}'] * data_size))
    return build_ordered_prompt_data


@pytest.fixture
def mock_leopard_data():
    def build_leopard_data(data_path, dataset_name, data_size=1500):
        leopard_dir = data_path / "leopard"
        leopard_dir.mkdir()
        dataset_dir = leopard_dir / dataset_name
        dataset_dir.mkdir()
        dataset = dataset_dir / f"{dataset_name}_eval.json"
        dataset.write_text(json.dumps([{"sentence1": "some sentence", "label": "0"}] * data_size))
    return build_leopard_data


def test_init_dataset_subj(tmp_path, mock_ordered_prompt_data):
    mock_ordered_prompt_data(tmp_path, "subj")
    dataset = init_dataset("subj", 42, tmp_path)
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is subjective or objective."
    )
    assert dataset.dataset_name == "subj"
    assert dataset.output_classes.protos == ["subjective", "objective"]
    train_sentences, train_label = dataset.get_data("train")
    assert "train sentence\nOptions:\n- subjective\n- objective" in train_sentences
    assert "0" in train_label
    dev_sentences, dev_label = dataset.get_data("dev")
    assert "dev sentence\nOptions:\n- subjective\n- objective" in dev_sentences
    assert "1" in dev_label
    test_sentences, test_label = dataset.get_data("test")
    assert "test sentence\nOptions:\n- subjective\n- objective" in test_sentences
    assert "2" in test_label


def test_init_dataset_mpqa(tmp_path, mock_ordered_prompt_data):
    mock_ordered_prompt_data(tmp_path, "mpqa")
    dataset = init_dataset("mpqa", 42, tmp_path)
    assert (
        dataset.instruction
        == "Read the following review, then choose whether it is negative or positive."
    )
    assert dataset.dataset_name == "mpqa"
    assert dataset.output_classes.protos == ["negative", "positive"]


def test_init_dataset_trec(tmp_path, mock_ordered_prompt_data):
    mock_ordered_prompt_data(tmp_path, "trec")
    dataset = init_dataset("trec", 42, tmp_path)
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


def test_init_dataset_disaster(tmp_path, mock_leopard_data):
    mock_leopard_data(tmp_path, "disaster")
    dataset = init_dataset("disaster", 42, tmp_path)
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is relevant to a disaster."
    )
    assert dataset.dataset_name == "disaster"
    assert dataset.output_classes.protos == ["no", "yes"]
    assert dataset.train_size == 1000
    assert dataset.dev_size == 250
    assert dataset.test_size == 250


def test_init_dataset_airline(tmp_path, mock_leopard_data):
    mock_leopard_data(tmp_path, "airline")
    dataset = init_dataset("airline", 42, tmp_path)
    assert (
        dataset.instruction
        == "Read the following sentence, then choose whether it is positive, negative, or neutral."
    )
    assert dataset.dataset_name == "airline"
    assert dataset.output_classes.protos == ["positive|Positive", "negative|Negative", "neutral|Neutral"]
    assert dataset.train_size == 1000
    assert dataset.dev_size == 250
    assert dataset.test_size == 250


def test_init_dataset_hyperbaton(tmp_path, mock_bbh_data):
    mock_bbh_data(tmp_path, "hyperbaton")
    dataset = init_dataset("hyperbaton", 42, tmp_path)
    assert dataset.instruction == "Which sentence has the correct adjective order:"
    assert dataset.dataset_name == "hyperbaton"
    assert dataset.output_classes.protos == ["a|A", "b|B"]
    assert dataset.train_size == 5
    assert dataset.dev_size == 5
    assert dataset.test_size == 10
    # shuffle options
    train_sentences, _ = dataset.get_data("train")
    assert "Another sentence:\nOptions:\n(A) option A\n(B) option B" in train_sentences
    assert "Another sentence:\nOptions:\n(A) option B\n(B) option A" in train_sentences


def test_init_dataset_navigate(tmp_path, mock_bbh_data):
    mock_bbh_data(tmp_path, "navigate")
    dataset = init_dataset("navigate", 42, tmp_path)
    assert (
        dataset.instruction
        == "Read the following sentence, then determine whether you return to the starting point."
    )
    assert dataset.dataset_name == "navigate"
    assert dataset.output_classes.protos == ["yes|Yes", "no|No"]
    assert dataset.train_size == 5
    assert dataset.dev_size == 5
    assert dataset.test_size == 10
    # do not shuffle options
    train_sentences, _ = dataset.get_data("train")
    assert 'Another sentence:\nOptions:\n(A) option A\n(B) option B' in train_sentences
    assert 'Another sentence:\nOptions:\n(A) option B\n(B) option A' not in train_sentences


def test_init_dataset_date_understanding(tmp_path, mock_bbh_data):
    mock_bbh_data(tmp_path, "date_understanding")
    dataset = init_dataset("date_understanding", 42, tmp_path)
    assert dataset.instruction == "Infer the date from context."
    assert dataset.dataset_name == "date_understanding"
    assert dataset.output_classes.protos == ["a|A", "b|B", "c|C", "d|D", "e|E", "f|F"]
    assert dataset.train_size == 5
    # for date understanding, we add training data to dev set, as the dev set is too small
    assert dataset.dev_size == 10
    assert dataset.test_size == 10


def test_init_dataset_logical_deduction_seven_objects(tmp_path, mock_bbh_data):
    mock_bbh_data(tmp_path, "logical_deduction_seven_objects")
    dataset = init_dataset("logical_deduction_seven_objects", 42, tmp_path)
    assert (
        dataset.instruction
        == "A seven-object logical deduction task which requires deducing the order of a sequence of objects."
    )
    assert dataset.dataset_name == "logical_deduction_seven_objects"
    assert dataset.output_classes.protos == ["a|A", "b|B", "c|C", "d|D", "e|E", "f|F", "g|G"]


def test_init_dataset_gsm8k(tmp_path):
    data_size = 100
    dataset = HFDataset.from_dict({
        "question": [f"question {i}" for i in range(data_size)],
        "answer": [f"answer #### {i}" for i in range(data_size)],
    })
    dataset_splits = dataset.train_test_split(test_size=0.4)
    dataset_dict = HFDatasetDict({  
        'train': dataset_splits['train'],  
        'test': dataset_splits['test'],
    })
    with patch("dln.dataset.hf_load_dataset", MagicMock(return_value=dataset_dict)):
        dataset = init_dataset("gsm8k", 42, tmp_path)
    train_sentences, train_labels = dataset.get_data("train")
    assert all([re.match(r"question \d+", s) for s in train_sentences])
    assert all([re.match(r"\d+", l) for l in train_labels])
    assert dataset.train_size == 30

    dev_sentences, dev_labels = dataset.get_data("dev")
    assert all([re.match(r"question \d+", s) for s in dev_sentences])
    assert all([re.match(r"\d+", l) for l in dev_labels])
    assert dataset.dev_size == 30

    test_sentences, test_labels = dataset.get_data("test")
    assert all([re.match(r"question \d+", s) for s in test_sentences])
    assert all([re.match(r"\d+", l) for l in test_labels])
    assert dataset.test_size == 40


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
