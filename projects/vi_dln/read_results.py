import glob
import sys
import os
import pandas as pd
import re
import json
import numpy as np
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

    if init_dev_accuracy:
        results[name]["init_dev"].append(init_dev_accuracy[0])
    if dev_accuracy:
        results[name]["dev"].append(np.max(dev_accuracy))
    if test_accuracy:
        results[name]["test"].append(test_accuracy[0])
    if token_cost:
        results[name]["cost"].append(token_cost[0])


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


data = []
for k, v in results.items():
    data.append(
        {
            "name": k,
            "init_dev": np.mean(v["init_dev"]) if "init_dev" in v else None,
            "dev": np.mean(v["dev"]),
            "test": np.mean(v["test"]),
            "cost": np.mean(v["cost"]),
            "dstd": np.std(v["dev"]),
            "tstd": np.std(v["test"]),
            "tcf": mean_confidence_interval(v["test"]),
            "seeds": len(v["dev"])
        }
    )


print(pd.DataFrame.from_records(data).sort_values(by="dev", ascending=True).to_markdown())
