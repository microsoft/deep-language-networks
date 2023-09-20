import glob
import os
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


files = glob.glob(root + "/**/log.txt", recursive=True)
for method in files:
    # time should be -2, method -3
    with open(os.path.dirname(method) + "/config.json", "r") as f:
        args = json.loads(f.read())

    name = "/".join(method.split("/")[-3:-2])
    if "-seed=" in name:
        name = name.replace(f"-seed={args['seed']}", "")

    print(name)

    if name not in results:
        results[name] = {"dev": [], "test": [], "cost": [], "args": None}
        results[name]['args'] = args

    dev_accuracy = []
    test_accuracy = []
    token_cost = []

    with open(method, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Best Dev Accuracy: " in line:
                line = escape_ansi(line).partition("Best Dev Accuracy: ")[-1]
                dev_accuracy.append(float(line.strip()))
            elif "Test Accuracy after training: " in line:
                line = escape_ansi(line).partition("Test Accuracy after training: ")[-1]
                test_accuracy.append(float(line.strip()))
            elif "Token cost: " in line:
                line = escape_ansi(line).partition("Token cost: ")[-1]
                token_cost.append(float(line.strip()))

    if dev_accuracy:
        results[name]["dev"].append(np.max(dev_accuracy))
    if test_accuracy:
        results[name]["test"].append(test_accuracy[0])
    if token_cost:
        results[name]["cost"].append(token_cost[0])


data = []
for key, val in results.items():
    res = {"name": key}
    for s in ["dev", "test", "cost"]:
        if val[s]:
            res[s] = "{:.3f}".format(np.mean(val[s])) + " +/- " + "{:.3f}".format(np.std(val[s]))
            res["n_seeds"] = len(val[s])
    data.append(res)


import pandas

pandas.set_option('display.max_colwidth', max(len(k) + 10 for k in results))

try:
    print(pandas.DataFrame(data).sort_values("dev"))
except:
    print(pandas.DataFrame(data))
