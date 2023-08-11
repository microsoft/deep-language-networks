import glob
import sys
import re
import json
import numpy as np
from termcolor import colored


def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


root = sys.argv[1]
results = {}

for method in glob.glob(root + "/**/output.log", recursive=True):
    # time should be -2, method -3
    name = "/".join(method.split("/")[-3:-2])
    if name not in results:
        results[name] = {"dev": [], "test": [], "cost": [], "args": None}

    dev_accuracy = []
    test_accuracy = []
    token_cost = []
    with open(method, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                args = line[line.find("{") :]
                json_args = json.loads(args)
                results[name]["args"] = json_args
            elif "BEST DEV ACC:" in line:
                line = escape_ansi(line).partition("BEST DEV ACC:")[-1]
                dev_accuracy.append(float(line.strip()))
            elif "TEST ACC:" in line:
                line = escape_ansi(line).partition("TEST ACC:")[-1]
                test_accuracy.append(float(line.strip()))
            elif "COST:" in line:
                line = escape_ansi(line).partition("COST:")[-1]
                token_cost.append(float(line.strip()))

    if dev_accuracy:
        results[name]["dev"].append(np.max(dev_accuracy))
    if test_accuracy:
        results[name]["test"].append(test_accuracy[0])
    if token_cost:
        results[name]["cost"].append(token_cost[0])

data = []
for key, val in results.items():
    results = {"name": key}
    for s in ["dev", "test"]:
        if val[s]:
            results[s] = "{:.3f}".format(np.mean(val[s])) + " +/- " + "{:.3f}".format(np.std(val[s]))
            results["n_seeds"] = len(val[s])
    data.append(results)


import pandas

print(pandas.DataFrame(data).sort_values("dev"))
