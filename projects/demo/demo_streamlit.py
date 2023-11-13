import argparse
import json
import textwrap

import altair as alt
import streamlit as st
import numpy as np
import pandas as pd
from jinja2 import Template


forward_template_L1 = Template(
    "{{ input }}\n\n{{ prompt }} Let's think step by step."
)  # Loaded template suffix_forward_tbs v1.0
forward_template_L2 = Template(
    "{{ prompt }}\n\n{{ input }}\n\nAnswer:"
)  #  Loaded template classify_forward v3.0

DATASETS = [
    ("subj", "1 Layer - Subj"),
    ("hyperbaton", "1 Layer - Hyperbaton"),
    ("navigate", "2 Layers - Navigate"),
]


def wrap_text(text, width=100):
    text = text.replace("\n\n", "\n")
    return "\n".join("\n".join(textwrap.wrap(line, width)) for line in text.split("\n"))


def load_data(log_file, dataset):
    with open(log_file) as f:
        logs = json.load(f)[dataset]

    flattened_data = []
    flattened_candidates = []
    for item in logs["training"]:
        flat_item = {"step": item["step"]}
        flat_item.update(
            {
                metric: value if value is not None else np.nan
                for metric, value in item["metrics"].items()
            }
        )
        flat_item.update(
            {f"layer_{i}": wrap_text(l) for i, l in enumerate(item["layers"], 1)}
        )
        flattened_data.append(flat_item)
        candidates_data = {}
        for layer, candidates in enumerate(item["candidates"], 1):
            for idx, candidate in enumerate(candidates):
                candidate_data = candidates_data.setdefault(idx, {"step": item["step"]})
                candidate_data[f"Layer {layer} candidate"] = candidate["layer"]
                candidate_data[f"Layer {layer} score"] = candidate["score"]
        flattened_candidates += list(candidates_data.values())

    flattened_examples = []
    for i, example in enumerate(logs["examples"]):
        for item in example["trace"]:
            flat_item = {
                "id": i + 1,
                "input": wrap_text(example["input"]),
                "label": example["label"],
                "step": item["step"],
                "hidden": wrap_text(item["hiddens"][0]) if item["hiddens"] else "",
                "output": wrap_text(item["output"]),
            }
            flattened_examples.append(flat_item)

    return (
        pd.DataFrame(flattened_data),
        pd.DataFrame(flattened_candidates).dropna(),
        pd.DataFrame(flattened_examples),
    )


def load_dataset_names(log_file):
    with open(log_file) as f:
        logs = json.load(f)
    return [(x, x) for x in list(logs.keys())]


def main(args):
    datasets = load_dataset_names(args.logfile) if args.logfile else DATASETS
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; margin-bottom: 80px'>Deep Language Networks</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # find navigate dataset index or default to 0
        selectbox_index = next((i for i, (dataset_id, _) in enumerate(datasets)if dataset_id == 'navigate'), 0)
        dataset_selectbox = st.selectbox("Dataset", datasets, index=selectbox_index, format_func=lambda x: x[1])
        dataset_selectbox = dataset_selectbox[0]
        df, candidates, examples = load_data(
            args.logfile or "data.json", dataset_selectbox
        )

        highlight_example = st.selectbox("Example", [i for i in range(20)], format_func=lambda x: x + 1)
        highlight_step = st.slider("Step", 1, 20)

        st.write("")
        table_data = []
        table_data.append(f"| **Input:** | {examples[examples['step'] == highlight_step]['input'].iloc[highlight_example]}")
        table_data.append(f"| **Layer 1 prompt:** | {df[df['step'] == highlight_step]['layer_1'].values[0]}")
        if 'layer_2' in df.columns:
            table_data.append(f"| **Hidden:** | {examples[examples['step'] == highlight_step]['hidden'].iloc[highlight_example]}")
            table_data.append(f"| **Layer 2 prompt:** | {df[df['step'] == highlight_step]['layer_2'].values[0]}")
        table_data.append(f"| **Output:** | {examples[examples['step'] == highlight_step]['output'].iloc[highlight_example]}")
        table_data.append(f"| **Label:** | {examples[examples['step'] == highlight_step]['label'].iloc[highlight_example]}")
        table_data = [x.replace('\n', '<br>') for x in table_data]
        table_data_str = "\n".join(table_data)

        st.markdown(f"\n| | |\n| --- | --- |\n{table_data_str}", unsafe_allow_html=True)
        st.write("")

    with col2:
        melted_df = df.melt(id_vars=['step'], value_vars=['acc', 'run_acc'], var_name='metric', value_name='value')
        melted_df['metric'] = melted_df['metric'].replace(['acc', 'run_acc'], ['Batch', 'Run Avg'])
        combined_chart = alt.Chart(melted_df).mark_line().encode(
            y=alt.Y('value:Q', title="accuracy", scale=alt.Scale(domain=[0.4, 1.0])),
            x='step:Q',
            color=alt.Color(
                'metric:N',
                scale=alt.Scale(domain=['Batch', 'Run Avg'], range=['steelblue', 'lightblue']),
                legend=alt.Legend(title="Train Accuracy")
            ),
        )

        # # Highlight a specific step
        # highlight_step = 10
        # # Add a selection
        # highlight_step = alt.selection_single(fields=['step'], on='click', nearest=True, init={'step': 10}, empty='none')

        # # Add a vertical rule at the specific step
        highlight_rule = alt.Chart(pd.DataFrame({'step': [highlight_step]})).mark_rule(color='red').encode(x='step:Q')
        # highlight_rule = alt.Chart().mark_rule(color='red').encode(x='step:Q').transform_filter(highlight_step)

        # # Combine the line chart, vertical rule, and text label
        alt_acc = alt.layer(
            combined_chart, highlight_rule, data=melted_df
        ).properties(height=500)

        st.altair_chart(alt_acc, use_container_width=True)

        activate_elbo = st.toggle("Elbo")
        elbo = df[["run_elbo"]]
        if activate_elbo:
            st.line_chart(elbo, height=500)

    prompt_candidates = st.toggle("Prompt Candidates")
    if prompt_candidates:
        # list all columns from candidates dataframe except the 'step' columns
        cols = [col for col in candidates.columns if col != 'step']
        st.dataframe(
            candidates[candidates["step"] == highlight_step][cols],
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", nargs="?", help="Log file to use (JSON).")
    args = parser.parse_args()

    main(args)
