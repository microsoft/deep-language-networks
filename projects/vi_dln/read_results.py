import glob
import sys
import os
import re
import json
import numpy as np


def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


root = sys.argv[1]
results = {}

for method in glob.glob(root + "/**/output.log", recursive=True):
    # time should be -2, method -3
    name = method.split("/")[-3]
    if name not in results:
        results[name] = {"dev": [], "test": [], "args": None}

    dev_accuracy = []
    test_accuracy = []
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

    if dev_accuracy:
        results[name]["dev"].append(np.max(dev_accuracy))
    if test_accuracy:
        results[name]["test"].append(test_accuracy[0])


for key, val in results.items():
    for s in ["dev", "test"]:
        if val[s]:
            print(
                "{:50s}".format(key),
                "--",
                s,
                "--",
                val[s],
                "AVG: ",
                np.mean(val[s]),
            )
