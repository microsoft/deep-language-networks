import glob
import sys
import re
import json
from itertools import combinations

import pandas as pd
import numpy as np
import scipy.stats



def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[\[0-9:;<=>?\]]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


root = sys.argv[1]
results = {}

for method in glob.glob(root + "/**/output.log", recursive=True):
    # time should be -2, method -3
    name = "/".join(method.split("/")[:-2])
    if name not in results:
        results[name] = {"dev": [], "test": [], "cost": [], "args": None, "init_dev": []}

    dev_accuracy = []
    test_accuracy = []
    token_cost = []
    init_dev_accuracy = []

    with open(method, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                args = line[line.find("{") :]
                json_args = json.loads(args)
                results[name]["args"] = json_args
            elif "INIT DEV ACC:" in line:
                line = escape_ansi(line).partition("INIT DEV ACC:")[-1]
                init_dev_accuracy.append(float(line.strip()))
            elif "BEST DEV ACC:" in line:
                line = escape_ansi(line).partition("BEST DEV ACC:")[-1]
                dev_accuracy.append(float(line.strip()))
            elif "TEST ACC:" in line:
                line = escape_ansi(line).partition("TEST ACC:")[-1]
                test_accuracy.append(float(line.strip()))
            elif "COST:" in line:
                line = escape_ansi(line).partition("COST:")[-1]
                token_cost.append(float(line.strip()))

    # skip jobs not completed
    if dev_accuracy and test_accuracy:
        results[name]["dev"].append(np.max(dev_accuracy))
        results[name]["test"].append(test_accuracy[0])
        results[name]["init_dev"].append(init_dev_accuracy[0] if init_dev_accuracy else np.nan)
        results[name]["cost"].append(token_cost[0] if token_cost else np.nan)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def top_k(data, k):
    """
    Generate all combinations of k seeded-runs
    Take the argmax dev score for each combination
    Report mean and std of the test score over all combinations (given their argmax).

    Conceptually, in the case of top-1, it is simply the average of all the
    runs and for top-10 (given 10 seeds total) it is equivalent to taking
    the argmax over all the seeds and reporting its test score.
    """
    test_scores = np.array(data['test'])
    dev_scores = np.array(data['dev'])

    if k <= 0 or k > len(test_scores):
        return np.nan, np.nan

    indices = np.arange(len(dev_scores))
    combination_indices = combinations(indices, k)

    max_test_scores = []
    for c in combination_indices:
        argmax_dev_index = np.argmax(dev_scores[list(c)])
        max_test_scores.append(test_scores[list(c)[argmax_dev_index]])

    mean_test_score = np.mean(max_test_scores)
    std_test_score = np.std(max_test_scores)

    return mean_test_score, std_test_score

def find_dataset_name(dataset_path):
    dataset_names = ["navigate", "hyperbaton", "date_understanding", "logical_deduction_seven_objects", "disaster", "airline", "mpqa", "trec", "subj"]
    for name in dataset_names:
        if name in dataset_path:
            return name.split("_")[0]

data = []
for k, v in results.items():
    top_k_1_mean_test, top_k_1_std_test = top_k(v, 1)
    top_k_3_mean_test, top_k_3_std_test = top_k(v, 3)
    top_k_argmax_mean_test, _ = top_k(v, len(v["test"]))

    # TODO: receive min_seeds as parameter
    if len(v["dev"]) <= 2:
        continue
    data.append(
        {
            # "name": k,
            # "name": k[:30],
            "name": find_dataset_name(k),
            "seeds": len(v["dev"]),
            "init_dev": np.mean(v["init_dev"]) if "init_dev" in v else None,
            "dev": np.mean(v["dev"]),
            "test": np.mean(v["test"]),
            "cost": np.mean(v["cost"]),
            "dstd": np.std(v["dev"]),
            "tstd": np.std(v["test"]),
            "tcf": mean_confidence_interval(v["test"]),
            "top_k_1_mean_test": top_k_1_mean_test,
            "top_k_1_std_test": top_k_1_std_test,
            "top_k_3_mean_test": top_k_3_mean_test,
            "top_k_3_std_test": top_k_3_std_test,
            "top_k_argmax_mean_test": top_k_argmax_mean_test,
            "log_path": k,
        }
    )

print(pd.DataFrame.from_records(data).sort_values(by=["name", "dev"], ascending=[True, False]).to_markdown(index=False))
