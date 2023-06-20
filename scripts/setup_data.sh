#!/bin/bash

# Create data folder
mkdir -p data

# Get Ordered Prompt data
wget https://github.com/yaolu/Ordered-Prompt/archive/refs/heads/main.zip
unzip main.zip
mv Ordered-Prompt-main/data data/ordered_prompt
rm -rf Ordered-Prompt-main
rm -f main.zip

# Get Leopard data
wget https://github.com/iesl/leopard/archive/refs/heads/master.zip
unzip master.zip
mv leopard-master/data/json data/leopard
rm -rf leopard-master
rm -f master.zip

# Get BBH data
wget https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip
unzip main.zip
mv BIG-Bench-Hard-main/bbh data/
rm -rf BIG-Bench-Hard-main
rm -f main.zip

# Preprocess BBH data removing points from BigBench to avoid data contamination
python scripts/split_bigbench_date_understanding.py
python scripts/split_bigbench_hyperbaton.py
python scripts/split_bigbench_logical_deduction_seven_objects.py
python scripts/split_bigbench_navigate.py