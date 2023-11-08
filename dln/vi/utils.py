import logging
import os
import copy
import json
from typing import Optional

import numpy as np


# Use this function to log messages to the file
def log_message(*messages):
    print(*messages)
    logging.info(" ".join(map(str, messages)))


def compute_pairwise_kl(lps):
    """We make updates constrained by the KL divergence between the function induced by the current prompt
    and the function induced by the new prompt. We compute the KL divergence
    between the functions induced by the prompts.

    The distribution induced by the current prompt is the first element of the second axis in lps.
    """
    # compute pairwise kl, considers reference always as the first prompt
    return (
        (lps[:, :1, :] * (np.log(lps[:, :1, :]) - np.log(lps[:, :, :]))).sum(-1).mean(0)
    )

class ResultLogEntry():
    def __init__(self):
        self.hiddens = None
        self.candidates = [[], []]
        self.metrics = {}
        self.outputs = []

    def log_metric(self, metric: str, value: Optional[float]):
        if value is not None:
            value = float(value)

        self.metrics[metric] = value

    def log_outputs(self, outputs):
        self.outputs = outputs

    def log_hiddens(self, hiddens, size):
        self.hiddens = [[]] * size if hiddens is None else [[h] for h in hiddens]

    def log_candidates(self, p_tilde_2, p2_elbo, p_tilde_1=None, p1_elbo=None):
        """
            If one_layer, p_tilde_1 and p1_elbo are None,
            and we only store the two-layer candidates in the 0th list element. 1st element stays [].
            If two_layer, we store the first layer candidates in the 0th list element
            and the second layer candidates in the 1st list element.
        """
        self.candidates = [[],[]]
        if p_tilde_1 is not None:
            for i in range(p_tilde_1.shape[0]):
                self.candidates[0].append({
                    "layer": p_tilde_1[i],
                    "score": float(p1_elbo[i]),
                })
            p2_ind = 1
        else:
            p2_ind = 0
        for i in range(p_tilde_2.shape[0]):
            self.candidates[p2_ind].append({
                "layer": p_tilde_2[i],
                "score": float(p2_elbo[i]),
            })


class ResultLogWriter(object):
    def __init__(self, name: str, path: str):
        """
        Args:
            name: File name
            path: File location
        Returns:
            A ResultLogWriter object
        """
        self.name = name
        self.path = path
        self.result_dict = {}
        self.result_dict[self.name] = {'training': [], 'examples': []}

    def write_result(self, step, layers, metrics, candidates):
        self.result_dict[self.name]['training'].append({'step': step})
        self.result_dict[self.name]['training'][-1]['layers'] = copy.deepcopy(layers)
        self.result_dict[self.name]['training'][-1]['metrics'] = copy.deepcopy(metrics)
        self.result_dict[self.name]['training'][-1]['candidates'] = copy.deepcopy(candidates)

    def write_examples(self, step, inputs, labels, outputs, hiddens):
        """
        Args:
            step: The iteration number
            inputs: A list of input strings
            labels: A list of label strings
            outputs: A list of output strings
            hiddens: A list of hidden strings for two-layer-dlns
        An element of the "examples" list in the json file looks like:
        {
            "input": "Do cats sit on mats?",
            "label": "Yes",
            "trace": [
                {
                    "step": 0,
                    "hiddens": ["Cats are picky."],
                    "output": "No"
                },
                {
                    "step": 1,
                    "hiddens": ["Cats would sit anywhere."],
                    "output": "Yes"
                }
            ]
        }
        """
        for inp, lab, outp, hidden in zip(inputs, labels, outputs, hiddens):
            # Get the element in the list of examples that matches the input
            example = next((ex for ex in self.result_dict[self.name]['examples'] if ex['input'] == inp), None)
            if example is None:
                self.result_dict[self.name]['examples'].append({
                    "input": inp,
                    "label": lab,
                    "trace": [{"step": step, "hiddens": hidden, "output": outp}],
                })
            else:
                example['trace'].append({"step": step, "hiddens": hidden, "output": outp})

    def save_to_json_file(self):
        # self.path is a path to a file
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, 'r') as f:
                print('Loading existing json file %s' % self.path)
                loaded_dict = json.load(f)
        except FileNotFoundError:
            loaded_dict = {}
        if self.name not in loaded_dict:
            # Append or add the json dictionary if the result doesn't exist
            loaded_dict[self.name] = self.result_dict[self.name]
            with open(self.path, 'w') as f:
                json.dump(loaded_dict, f, indent=4)
        else:
            print(f"Result named {self.name} already exists in {self.path}!")