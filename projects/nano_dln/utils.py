import datetime
import json
import logging
import os
import yaml
import copy
import random

import numpy as np
import torch


def get_start_time():
    start_time = str(datetime.datetime.now().strftime("%b %d %Y %H:%M:%S.%f")).replace(
        " ", "_"
    )
    return start_time


def dumps_config(config, path, filename=None):
    if filename is None:
        filename = "config.json"
    if not filename.endswith(".json"):
        filename += ".json"
    with open(os.path.join(path, filename), "w") as f:
        json.dump(config, f)


def setup_logging(log_level, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "log.txt")),
            logging.StreamHandler(),
        ],
        format="%(asctime)-15s %(levelname)-8s %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.info(
        "New experiment, log will be at %s",
        os.path.join(log_dir, "log.txt"),
    )


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    return dict(keys)


def load_config(config_file):
    assert os.path.exists(config_file), "Invalid config file"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config


def is_in_debug_mode() -> bool:
    """Determine whether debug mode should be enabled.

    To enable debug mode, set the environment variable DLN_DEBUG to "1" or "true".
    """
    if "DLN_DEBUG" not in os.environ:
        return False
    env_value = os.environ["DLN_DEBUG"].lower()
    if env_value in {"1", "true"}:
        return True
    elif env_value in {"", "0", "false"}:
        return False
    else:
        raise ValueError(f'{env_value!r} is not a valid value for "DLN_DEBUG"')


DEBUG_MODE = is_in_debug_mode()


class Logger:
    def __init__(self):
        self.log_list = []

    def log(self, string, show=True):
        if not isinstance(string, str) or len(string) == 0:
            return
        self.log_list.append(string)
        if show:
            print(string)

    def get_log_list(self):
        return copy.deepcopy(self.log_list)
