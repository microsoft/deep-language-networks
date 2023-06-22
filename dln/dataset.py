import os
import json
import yaml
import numpy as np
from collections import defaultdict

from os.path import join as pjoin

from dln.score import OutputClasses


class Dataset:
    def __init__(self, dataset_path, dataset, seed, use_label_mapping=True, prefix=None, append_options=True):
        self.dataset_name = dataset
        self.data_path = dataset_path
        self.random_seed = seed
        self.dataset_info = self._load_config(
            pjoin(os.path.dirname(os.path.abspath(__file__)), "dataset_info.yaml")
        )
        self.prefix = self.dataset_info[self.dataset_name].get("prefix", "")
        self.label_mapping = self.dataset_info[self.dataset_name].get("label_mapping", {})
        self.use_label_mapping = (
            use_label_mapping
            and self.label_mapping is not None
        )
        self.append_options = append_options
        self.instruction = self.dataset_info[self.dataset_name]["instruction"]
        self.rng = np.random.RandomState(self.random_seed)

        # load dataset from file
        self.dataset = dict(
            train_per_class=dict(),
            train=dict(sentence=[], label=[]),
            dev=dict(sentence=[], label=[]),
            test=dict(sentence=[], label=[]),
        )
        self.load_dataset()

        print("loaded dataset from %s ..." % self.data_path)
        print(
            "we have %s training, %s dev, and %s test data points."
            % (self.train_size, self.dev_size, self.test_size)
        )
        self.reset()

    @staticmethod
    def _load_config(config_file):
        assert os.path.exists(config_file), "Invalid config file"
        with open(config_file) as reader:
            config = yaml.safe_load(reader)
        return config

    @property
    def train_size(self):
        return len(self.dataset["train"]["label"])

    @property
    def dev_size(self):
        return len(self.dataset["dev"]["label"])

    @property
    def test_size(self):
        return len(self.dataset["test"]["label"])

    def resize(self, split, size):
        indices = np.random.permutation(np.arange(len(self.dataset[split]["label"])))[
            :size
        ]
        self.dataset[split]["label"] = [
            self.dataset[split]["label"][i] for i in indices
        ]
        self.dataset[split]["sentence"] = [
            self.dataset[split]["sentence"][i] for i in indices
        ]

    def reset(self):
        self.train_pointer, self.dev_pointer, self.test_pointer = 0, 0, 0

    def load_dataset(self):
        data_shuffling_rng = np.random.RandomState(42)

        if "bbh" in self.data_path:
            assert self.dataset_name in (
                "logical_deduction_seven_objects",
                "hyperbaton",
                "navigate",
                "date_understanding",
            ), self.dataset_name
            train_valid_file_path = os.path.join(self.data_path.replace("bbh", "bb_minus_bbh"), self.dataset_name + ".json")
            with open(train_valid_file_path) as fin:
                data = json.load(fin)
                data = data["examples"]
                train_size = len(data) // 2
                dev_size = len(data) - train_size
                train_size = min(train_size, 1000)  # some dataset has too many data, don't want to have a too large train/valid size
                dev_size = min(dev_size, 1000)
                assert train_size > 0, train_size
                assert dev_size > 0, dev_size
                data_shuffling_rng.shuffle(data)
                for i in range(len(data)):
                    if i < train_size:
                        split = "train"
                    elif i < train_size + dev_size:
                        split = "dev"
                    else:
                        break
                    input, target = data[i]["input"], data[i]["target"]
                    if self.dataset_name == "date_understanding" and split == "train":
                        # for date understanding, we add training data to dev set, as the dev set is too small
                        self.dataset["dev"]["sentence"].append(input)
                        self.dataset["dev"]["label"].append(target)
                    self.dataset[split]["sentence"].append(input)
                    self.dataset[split]["label"].append(target)

            test_file_path = os.path.join(self.data_path, self.dataset_name + ".json")
            with open(test_file_path) as fin:
                data = json.load(fin)
                data = data["examples"]
                for i in range(len(data)):
                    input, target = data[i]["input"], data[i]["target"]
                    self.dataset["test"]["sentence"].append(input)
                    self.dataset["test"]["label"].append(target)

        elif "ordered_prompt" in self.data_path:
            assert self.dataset_name in (
                "mpqa",
                "trec",
                "subj"), self.dataset_name

            for split in ["train", "dev", "test"]:
                _split = "dev_subsample" if split == "dev" else split
                file_path = os.path.join(
                    self.data_path, self.dataset_name, _split + ".jsonl"
                )
                sentence_list, label_list = [], []

                with open(file_path) as fin:
                    for line in fin:
                        datapoint = json.loads(line)
                        sentence, label = datapoint["sentence"], datapoint["label"]
                        if self.append_options:
                            sentence = [sentence, "Options:"] + ["- " + item for item in list(self.label_mapping.values())]
                            sentence = "\n".join(sentence)
                        sentence_list.append(sentence)
                        label_list.append(label)

                if split == "train":
                    indices = data_shuffling_rng.choice(len(sentence_list), 1000, replace=False)
                    for idx in indices:
                        self.dataset[split]["sentence"].append(sentence_list[idx])
                        self.dataset[split]["label"].append(label_list[idx])
                elif split == "dev":
                    self.dataset[split]["sentence"] = sentence_list
                    self.dataset[split]["label"] = label_list
                elif split == "test":
                    indices = data_shuffling_rng.choice(len(sentence_list), 250, replace=False)
                    for idx in indices:
                        self.dataset[split]["sentence"].append(sentence_list[idx])
                        self.dataset[split]["label"].append(label_list[idx])

        elif "leopard" in self.data_path:
            assert self.dataset_name in (
                "disaster",
                "airline"), self.dataset_name
            file_path = os.path.join(
                self.data_path, self.dataset_name, self.dataset_name + "_eval.json"
            )
            sentence_list, label_list = [], []

            with open(file_path) as fin:
                data = json.load(fin)
                for i in range(len(data)):
                    sentence, label = data[i]["sentence1"], data[i]["label"]
                    if self.append_options:
                        sentence = [sentence, "Options:"] + ["- " + item for item in list(self.label_mapping.values())]
                        sentence = "\n".join(sentence)
                    sentence_list.append(sentence)
                    label_list.append(label)
            indices = data_shuffling_rng.choice(len(sentence_list), 1500, replace=False)
            for idx in indices[:1000]:
                self.dataset["train"]["sentence"].append(sentence_list[idx])
                self.dataset["train"]["label"].append(label_list[idx])
            for idx in indices[1000:1250]:
                self.dataset["dev"]["sentence"].append(sentence_list[idx])
                self.dataset["dev"]["label"].append(label_list[idx])
            for idx in indices[1250:]:
                self.dataset["test"]["sentence"].append(sentence_list[idx])
                self.dataset["test"]["label"].append(label_list[idx])
        else:
            raise NotImplementedError

        train_per_class = defaultdict(list)
        for index, label in enumerate(self.dataset["train"]["label"]):
            train_per_class[label].append(index)
        self.dataset["train_per_class"] = train_per_class

    def reset_pointer(self, split):
        if split == "train":
            self.train_pointer = 0
        elif split == "dev":
            self.dev_pointer = 0
        elif split == "test":
            self.test_pointer = 0

    def get_batch(self, split, batch_size, random_sample=False, balance=False):
        assert batch_size > 0
        assert split in ["train", "dev", "test"]
        if split == "train":
            pointer = self.train_pointer
            data_size = self.train_size
            self.train_pointer += batch_size
            if self.train_pointer >= data_size:
                self.train_pointer = 0
        elif split == "dev":
            pointer = self.dev_pointer
            data_size = self.dev_size
            self.dev_pointer += batch_size
            if self.dev_pointer >= data_size:
                self.dev_pointer = 0
        else:
            pointer = self.test_pointer
            data_size = self.test_size
            self.test_pointer += batch_size
            if self.test_pointer >= data_size:
                self.test_pointer = 0

        if random_sample is True:
            if balance is True:
                indices = []
                pick_order = self.rng.choice(
                    list(self.dataset["train_per_class"].keys()),
                    len(self.dataset["train_per_class"].keys()),
                    replace=False,
                )

                i = 0
                while len(indices) < batch_size:
                    indices += self.rng.choice(
                        self.dataset["train_per_class"][pick_order[i % len(pick_order)]], 1
                    ).tolist()
                    i += 1
            else:
                indices = self.rng.choice(data_size, batch_size, replace=False)
        else:
            start = pointer
            end = min(start + batch_size, data_size)
            indices = np.arange(start, end)

        sentence_list, label_list = [], []
        for idx in indices:
            sentence_list.append(self.dataset[split]["sentence"][idx])
            if self.prefix is not None:
                sentence_list[-1] = sentence_list[-1].replace(self.prefix, "")

            if self.use_label_mapping:
                label_mapping = self.label_mapping
                label_list.append(label_mapping[self.dataset[split]["label"][idx]])
            else:
                label_list.append(self.dataset[split]["label"][idx])

        return sentence_list, label_list

    def iterate(self, split, batch_size, random_sample=False):
        if split == "train":
            self.train_pointer = 0
        elif split == "dev":
            self.dev_pointer = 0
        else:
            self.test_pointer = 0
        while True:
            yield self.get_batch(split, batch_size, random_sample)

            if not random_sample:
                if split == "dev" and self.dev_pointer == 0:
                    return

                if split == "test" and self.test_pointer == 0:
                    return

    def get_size(self, split):
        return len(self.dataset[split]["label"])

    def get_data(self, split="train", indices=None):
        """ get all data from a split """
        assert split in self.dataset
        if indices is None:
            res_sentence = self.dataset[split]["sentence"]
            res_label = self.dataset[split]["label"]
        else:
            assert isinstance(indices, list) and len(indices) > 0
            res_sentence, res_label = [], []
            for idx in indices:
                res_sentence.append(self.dataset[split]["sentence"][idx])
                res_sentence.append(self.dataset[split]["label"][idx])
        return res_sentence, res_label


def init_dataset(dataset_id, seed):
    dataset_location = {
        "subj": "./data/ordered_prompt",
        "mpqa": "./data/ordered_prompt",
        "trec": "./data/ordered_prompt",
        "disaster": "./data/leopard",
        "airline": "./data/leopard",
        "hyperbaton": "./data/bbh",
        "navigate": "./data/bbh",
        "date_understanding": "./data/bbh",
        "logical_deduction_seven_objects": "./data/bbh",
    }

    assert dataset_id in dataset_location, f"Dataset {dataset_id} not found"

    dataset = Dataset(dataset_location[dataset_id], dataset_id, seed)
    val_examples = {"hyperbaton": 300}.get(dataset_id, -1)
    protos = {
        "hyperbaton": ["a|A", "b|B"],
        "navigate": ['yes|Yes', 'no|No'],
        "date_understanding": ['a|A', 'b|B', 'c|C', 'd|D', 'e|E', 'f|F'],
        "logical_deduction_seven_objects": ['a|A', 'b|B', 'c|C', 'd|D', 'e|E', 'f|F', 'g|G'],
    }.get(dataset_id, list(dataset.label_mapping.values()))
    output_classes = OutputClasses(protos=protos)
    return dataset, output_classes, val_examples
