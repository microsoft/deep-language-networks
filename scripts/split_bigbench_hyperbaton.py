import os
import json
import codecs

from utils import download_json_from_url


bb_raw_json_url = "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/hyperbaton/task.json"
bbh_file_path = "data/bbh/hyperbaton.json"
option_list = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)"]

bbh_sentence, bbh_label = [], []
bbh_dict = {}
with open(bbh_file_path) as fin:
    data = json.load(fin)
    data = data["examples"]

    for i in range(len(data)):
        input, target = data[i]["input"], data[i]["target"]
        bbh_sentence.append(input)
        bbh_label.append(target)
        assert input not in bbh_dict
        bbh_dict[input] = len(bbh_sentence) - 1

data = download_json_from_url(bb_raw_json_url)
data = data["examples"]
bb_sentence, bb_label = [], []
bb_len = 0
in_bbh = 0
for i in range(len(data)):
    bb_len += 1
    input, target = data[i]["input"], data[i]["target_scores"]
    candidates = input.split("Which sentence has the correct adjective order:")[-1].strip()
    input = "Which sentence has the correct adjective order:"
    candidates = [candidates.split("\"")[1].strip(), candidates.split("\"")[3].strip()]
    res_input = [input, "Options:"]
    res_target = ""
    for j, key in enumerate(target):
        if target[key] == 1:
            res_target = option_list[j]
        res_input.append(" ".join([option_list[j], candidates[j]]))
    assert len(res_target) > 0
    res_input = "\n".join(res_input)
    if res_input in bbh_dict:
        in_bbh += 1
        continue

    bb_sentence.append(res_input)
    bb_label.append(res_target)
assert in_bbh == len(bbh_dict), "%s, %s" % (str(in_bbh), str(len(bbh_dict)))

data = []
for i in range(len(bb_sentence)):
    data.append({"input": bb_sentence[i], "target": bb_label[i]})
print("there are %s data points in big bench" % str(bb_len))
print("there are %s data points in big bench hard" % str(len(bbh_dict)))
print("collected %s data points from big bench - big bench hard" % str(len(data)))

bb_minus_bbh_file_path = "data/bb_minus_bbh/"
print("writing data to ", bb_minus_bbh_file_path)
if not os.path.exists(bb_minus_bbh_file_path):
    os.makedirs(bb_minus_bbh_file_path)
with codecs.open(bb_minus_bbh_file_path + "/hyperbaton.json", 'w', encoding='utf-8') as json_file:
    json.dump({"examples": data}, json_file, ensure_ascii=False)
