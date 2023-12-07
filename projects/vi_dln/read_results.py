import glob
import sys
import os
import re
import json
import numpy as np
from termcolor import colored
import scipy.stats as st
import pandas


def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


# root = sys.argv[1]
results = {}

for root in sys.argv[1:]:
    for method in glob.glob(root + "/**/output.log", recursive=True):
        # time should be -2, method -3, dataset -4
        # from ipdb import set_trace; set_trace()
        name = "/".join(method.split("/")[-4:-2])
        name = "/".join(method.split("/")[:-2])

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def shorten(name):
    name = name.replace("logical_deduction_seven_objects", "logic7")

    if len(name) > 50:
        name = name[:25] + "..." + name[-25:]

    return name

common_key_prefix = "/".join(os.path.commonprefix([k.split("/") for k in results])) + "/"

data = []
for key, val in results.items():
    print(val['test'])
    res = {
        "name": shorten(key.replace(common_key_prefix, "")),
        "dev": f"{np.mean(val['dev']):.3f}",
        "cost": f"{np.mean(val['cost']):.1f}",
        "test": f"{np.mean(val['test'])*100:2.1f}",
        "std": f"{np.std(val['test'])*100:2.3f}",
        "95%-ci": f"{mean_confidence_interval(val['test'])[1]*100:2.3f}",
        "n_seeds": len(val['test']),
    }

    # for s in ["dev", "test", "cost"]:
    #     if val[s]:
    #         res[s] = f"{np.mean(val[s]):.3f} +/- {np.std(val[s]):.3f} [{mean_confidence_interval(val[s])[1]:.3f}]"
    #         #res[s] = f"{np.mean(val[s]):.3f} {{\\tiny$\pm${mean_confidence_interval(val[s])[1]:.3f}}}"
    #         res["n_seeds"] = len(val[s])
    data.append(res)


pandas.set_option('display.max_colwidth', max(len(k) + 90 for k in results))

data = pandas.DataFrame(data)

# print(data)

try:
    print(data.sort_values("name"))
except:
    print(data)


# # Print average cost across all datasets
# #print("Average cost:")
# #print(np.mean([float(d["cost"]) for d in data.to_dict("records") if "cost" in d]))

# Export latex
datasets = [
    "hyperbaton",
    "navigate",
    "date_understanding",
    "logical_deduction_seven_objects",
    "mpqa",
    "trec",
    "subj",
    "disaster",
    "airline",
]


latex = ""
for nb_shots in ["0shot", "5shot", "16shot", "DLN-1", "DLN-2"]:
    latex += f"\;{nb_shots} & "
    for dataset in datasets:
        found = False
        for key, val in results.items():
            if nb_shots not in key:
                continue

            if dataset not in key:
                continue


            #print(f"{key}: {np.mean(val['test']):.3f} {{\\tiny$\pm${mean_confidence_interval(val['test'])[1]:.3f}}}")
            latex += f"{np.mean(val['test'])*100:2.1f}{{\\tiny$\pm${mean_confidence_interval(val['test'])[1]*100:2.1f}}} & "

            #print(f"{key}: {np.mean(val['cost'])/1000:.1f} {{\\tiny$\pm${mean_confidence_interval(val['cost'])[1]/1000:.1f}}}")
            #latex += f"{np.mean(val['cost'])/1000:.1f}{{\\tiny$\pm${mean_confidence_interval(val['cost'])[1]/1000:1.1f}}} & "

            #print(f"{key}: {{\\tiny \\color{{gray}} ({np.mean(val['cost'])/1000:.1f}) }}")
            #latex += f"{{\\tiny \\color{{gray}} ({np.mean(val['cost'])/1000:.1f}k)}} & "
            found = True

        if not found:
            latex += f" - & "

    latex = latex[:-2] + "\\\\\n"

print(latex)
