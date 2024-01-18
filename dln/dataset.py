from abc import abstractmethod
import json
import os
from collections import defaultdict
from os.path import join as pjoin

import numpy as np
import yaml
from datasets import load_dataset as hf_load_dataset

from dln.score import OutputClasses
from dln.vi.utils import log_message


def option_shuffle(data_point, rng):
    import re

    pattern = r'\([A-Z]\)\s(.*)'
    letters = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)']

    input = data_point['input']
    target = data_point['target']

    if "\nOptions:\n" not in input:
        raise ValueError("Error detected in data point, Options not found.")

    input, _, options = input.partition('\nOptions:\n')
    options = options.strip().split("\n")
    options_text = [re.findall(pattern, option)[-1] for option in options]

    random_indices = rng.permutation(range(len(options)))
    target_index = letters.index(target)

    new_target = letters[list(random_indices).index(target_index)]
    new_options = [options_text[i] for i in random_indices]
    new_options = [f'{letter} {text}' for letter, text in zip(letters, new_options)]

    new_data_point = {}
    new_data_point['input'] = f'{input}\nOptions:\n' + '\n'.join(new_options)
    new_data_point['target'] = new_target

    return new_data_point


class Dataset:
    def __init__(
        self,
        dataset_path: str,
        dataset: str,
        seed: int,
        use_label_mapping: bool = True,
        append_options: bool = True,
        n_few_shots: int = -1,
        max_train_size: int = -1,
        max_dev_size: int = -1,
        max_test_size: int = -1,
    ):
        self.dataset_name = dataset
        self.data_path = dataset_path
        self.random_seed = seed
        self.n_few_shots = n_few_shots
        self.dataset_info = self._load_config(
            pjoin(os.path.dirname(os.path.abspath(__file__)), "dataset_info.yaml")
        ).get(self.dataset_name, {})
        self.label_mapping = self.dataset_info.get("label_mapping", {})
        self.use_label_mapping = use_label_mapping and self.label_mapping
        self.append_options = append_options
        self.instruction = self.dataset_info.get("instruction", "")
        protos = self.dataset_info.get('protos', list(self.label_mapping.values()))
        self.output_classes = OutputClasses(protos=protos) if protos else None

        self.rng = np.random.RandomState(self.random_seed)
        self.few_shot_rng = np.random.RandomState(self.random_seed)

        # load dataset from file
        self.dataset = dict(
            train_per_class=dict(),
            train=dict(sentence=[], label=[]),
            dev=dict(sentence=[], label=[]),
            test=dict(sentence=[], label=[]),
        )
        self.load_dataset()
        self.resize_and_compute_per_class(
            max_train_size, max_dev_size, max_test_size
        )
        self.reset()

        log_message("loaded dataset from %s ..." % self.data_path)
        log_message(
            "we have %s training, %s dev, and %s test data points."
            % (self.train_size, self.dev_size, self.test_size)
        )

    @abstractmethod
    def load_dataset(self):
        pass

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

    def _get_few_shots(self):
        if self.n_few_shots <= 0:
            return None

        indices = []
        pick_order = self.few_shot_rng.choice(
            list(self.dataset["train_per_class"].keys()),
            len(self.dataset["train_per_class"].keys()),
            replace=False,
        )

        i = 0
        while len(indices) < self.n_few_shots:
            indices += self.few_shot_rng.choice(
                self.dataset["train_per_class"][pick_order[i % len(pick_order)]],
                1,
            ).tolist()
            i += 1
        x = [self.dataset["train"]["sentence"][i] for i in indices]

        if self.use_label_mapping:
            label_mapping = self.label_mapping
            y = [label_mapping[self.dataset["train"]["label"][i]] for i in indices]
        else:
            y = [self.dataset["train"]["label"][i] for i in indices]
        return list(zip(x, y))

    def resize_and_compute_per_class(self, max_train_size, max_dev_size, max_test_size):
        for split, max_size in zip(
            ("train", "dev", "test"),
            (max_train_size, max_dev_size, max_test_size),
        ):
            resize_rng = self.rng if split == "train" else np.random.RandomState(42)
            split_per_class = f"{split}_per_class"
            per_class = defaultdict(list)
            for index, label in enumerate(self.dataset[split]["label"]):
                per_class[label].append(index)
            self.dataset[split_per_class] = per_class

            if max_size > 0:
                log_message(f"Cutting {split} dataset to {max_size} examples.")
                indices = []
                pick_order = resize_rng.choice(
                    list(self.dataset[split_per_class].keys()),
                    len(self.dataset[split_per_class].keys()),
                    replace=False,
                )

                i = 0
                while len(indices) < max_size:
                    indices += resize_rng.choice(
                        self.dataset[split_per_class][
                            pick_order[i % len(pick_order)]
                        ],
                        1,
                    ).tolist()
                    i += 1

                self.dataset[split]["sentence"] = [self.dataset[split]["sentence"][i] for i in indices]
                self.dataset[split]["label"] = [self.dataset[split]["label"][i] for i in indices]

                per_class = defaultdict(list)
                for index, label in enumerate(self.dataset[split]["label"]):
                    per_class[label].append(index)
                self.dataset[split_per_class] = per_class

    def reset(self):
        self.train_pointer, self.dev_pointer, self.test_pointer = 0, 0, 0

    def reset_pointer(self, split):
        if split == "train":
            self.train_pointer = 0
        elif split == "dev":
            self.dev_pointer = 0
        elif split == "test":
            self.test_pointer = 0

    def get_batch(self, split, batch_size, random_sample=False, balance=False, return_few_shot=True):
        if balance is True and random_sample is False:
            raise ValueError("Balance batch must be sampled randomly.")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if split not in ["train", "dev", "test"]:
            raise ValueError(f"Invalid split: {split}")

        batch_rng = np.random.RandomState(42)
        if split == "train":
            batch_rng = self.rng
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
                example_pools = {}
                for key in self.dataset[f"{split}_per_class"].keys():
                    example_pools[key] = batch_rng.permutation(
                        self.dataset[f"{split}_per_class"][key]
                )
                i = 0
                pick_order = batch_rng.permutation(list(self.dataset[f"{split}_per_class"].keys()))
                while len(indices) < batch_size:
                    current_key = pick_order[i % len(pick_order)]

                    if sum(map(len, example_pools.values())) == 0:
                        raise ValueError(f"Not enough examples to sample batch of size {batch_size}.")

                    if len(example_pools[current_key]) > 0:
                        indices.append(example_pools[current_key][0])
                        example_pools[current_key] = example_pools[current_key][1:]
                    i += 1
            else:
                indices = batch_rng.choice(data_size, batch_size, replace=False)
        else:
            start = pointer
            end = min(start + batch_size, data_size)
            indices = np.arange(start, end)

        sentence_list, label_list = [], []
        for idx in indices:
            sentence_list.append(self.dataset[split]["sentence"][idx])

            if self.use_label_mapping:
                label_mapping = self.label_mapping
                label_list.append(label_mapping[self.dataset[split]["label"][idx]])
            else:
                label_list.append(self.dataset[split]["label"][idx])

        if not return_few_shot:
            return sentence_list, label_list

        few_shots = self._get_few_shots()
        return sentence_list, label_list, few_shots

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

    def get_data(self, split, indices=None):
        """get all data from a split"""
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


def init_dataset(
    dataset_id,
    seed,
    data_dir,
    n_few_shots=-1,
    max_train_size=-1,
    max_dev_size=-1,
    max_test_size=-1,
):
    datasets = {
        "subj": OrderedPrompt,
        "mpqa": OrderedPrompt,
        "trec": OrderedPrompt,
        "disaster": Leopard,
        "airline": Leopard,
        "hyperbaton": BBH,
        "navigate": BBH,
        "date_understanding": BBH,
        "logical_deduction_seven_objects": BBH,
        "gsm8k": GSM8K,
    }
    try:
        dataset_class = datasets[dataset_id]
    except KeyError:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset = dataset_class(
        dataset_path=data_dir,
        dataset=dataset_id,
        seed=seed,
        n_few_shots=n_few_shots,
        max_train_size=max_train_size,
        max_dev_size=max_dev_size,
        max_test_size=max_test_size,
    )
    return dataset


class BBH(Dataset):

    def load_dataset(self):
        self.data_path = os.path.join(self.data_path, "bbh")
        data_shuffling_rng = np.random.RandomState(42)
        train_valid_file_path = os.path.join(
            self.data_path.replace("bbh", "bb_minus_bbh"),
            self.dataset_name + ".json",
        )
        with open(train_valid_file_path) as fin:
            data = json.load(fin)

        data = data["examples"]
        train_size = len(data) // 2
        dev_size = len(data) - train_size
        train_size = min(
            train_size, 1000
        )  # some dataset has too many data, don't want to have a too large train/valid size
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

            # option shuffling in all multiple options cases, skip navigate (yes/no)
            if self.dataset_name != "navigate":
                data[i] = option_shuffle(data[i], data_shuffling_rng)

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

        # TODO: use a parameter instead
        if self.dataset_name == "hyperbaton":
            val_examples = 300
            self.dataset["dev"]["sentence"] = self.dataset["dev"]["sentence"][:val_examples]
            self.dataset["dev"]["label"] = self.dataset["dev"]["label"][:val_examples]


class Leopard(Dataset):

    def load_dataset(self):
        self.data_path = os.path.join(self.data_path, "leopard")
        data_shuffling_rng = np.random.RandomState(42)
        assert self.dataset_name in ("disaster", "airline"), self.dataset_name
        file_path = os.path.join(
            self.data_path, self.dataset_name, self.dataset_name + "_eval.json"
        )
        sentence_list, label_list = [], []

        with open(file_path) as fin:
            data = json.load(fin)
            for i in range(len(data)):
                sentence, label = data[i]["sentence1"], data[i]["label"]
                if self.append_options:
                    sentence = [sentence, "Options:"] + [
                        "- " + item for item in list(self.label_mapping.values())
                    ]
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


class OrderedPrompt(Dataset):

    def load_dataset(self):
        self.data_path = os.path.join(self.data_path, "ordered_prompt")
        data_shuffling_rng = np.random.RandomState(42)
        assert self.dataset_name in ("mpqa", "trec", "subj"), self.dataset_name

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
                        sentence = [sentence, "Options:"] + [
                            "- " + item
                            for item in list(self.label_mapping.values())
                        ]
                        sentence = "\n".join(sentence)
                    sentence_list.append(sentence)
                    label_list.append(label)

            if split == "train":
                indices = data_shuffling_rng.choice(
                    len(sentence_list), 1000, replace=False
                )
                for idx in indices:
                    self.dataset[split]["sentence"].append(sentence_list[idx])
                    self.dataset[split]["label"].append(label_list[idx])
            elif split == "dev":
                self.dataset[split]["sentence"] = sentence_list
                self.dataset[split]["label"] = label_list
            elif split == "test":
                indices = data_shuffling_rng.choice(
                    len(sentence_list), 250, replace=False
                )
                for idx in indices:
                    self.dataset[split]["sentence"].append(sentence_list[idx])
                    self.dataset[split]["label"].append(label_list[idx])


class GSM8K(Dataset):

    def __init__(
        self,
        dataset_path: str,
        dataset: str,
        seed: int,
        use_label_mapping: bool = False,
        append_options: bool = False,
        n_few_shots: int = -1,
        max_train_size: int = -1,
        max_dev_size: int = -1,
        max_test_size: int = -1,
    ):
        if use_label_mapping or append_options:
            log_message("GSM8K does not support label mapping or append options.")


        super().__init__(
            dataset_path=dataset_path,
            dataset=dataset,
            seed=seed,
            use_label_mapping=False,
            append_options=False,
            n_few_shots=n_few_shots,
            max_train_size=max_train_size,
            max_dev_size=max_dev_size,
            max_test_size=max_test_size,
        )

    @staticmethod
    def _split_answer(answers):
        steps = []
        final_answers = []
        for answer in answers:
            s, a = answer.split("#### ")
            steps.append(s)
            final_answers.append(a)
        return steps, final_answers

    def load_dataset(self):
        self.data_path = os.path.join(self.data_path, "huggingface/datasets")
        hf_dataset = hf_load_dataset(
            self.dataset_name, 'main', cache_dir=self.data_path
        ).shuffle(seed=42)

        train_size = hf_dataset["train"].num_rows // 2
        train_dataset = hf_dataset["train"][:train_size]
        dev_dataset = hf_dataset["train"][train_size:]
        test_dataset = hf_dataset['test']

        self.dataset["train"]["sentence"] = train_dataset['question']
        self.dataset["train"]["label"] = self._split_answer(train_dataset['answer'])[1]
        self.dataset["dev"]["sentence"] = dev_dataset['question']
        self.dataset["dev"]["label"] = self._split_answer(dev_dataset['answer'])[1]

        self.dataset["test"]["sentence"] = test_dataset['question']
        self.dataset["test"]["label"] = self._split_answer(test_dataset['answer'])[1]
