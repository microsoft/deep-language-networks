import argparse
import json
import textwrap

import streamlit as st

import altair as alt
import numpy as np
import pandas as pd
from jinja2 import Template

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
                candidate_data[f"layer_{layer}_candidate"] = candidate["layer"]
                candidate_data[f"layer_{layer}_score"] = candidate["score"]
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
    example_dropdown = 0
    df, candidates, examples = load_data(
        args.logfile or "data.json", dataset_id
    )

    # Merge layers and examples
    df = df.merge(examples, on="step", how="left")

    EXAMPLE_ID = example_dropdown or 1
    dev_df = df
    dev_df = df[df["id"] == EXAMPLE_ID]
    dev_df = dev_df[dev_df["dev_acc"] >= 0]

    NB_LAYERS = len([c for c in dev_df.columns if c.startswith("layer")])

    if example_dropdown == 0:
        if NB_LAYERS == 1:
            layers_columns = [c for c in dev_df.columns if c.startswith("layer")]
            for column in layers_columns:
                # Wrap text for display in hover
                dev_df[column] = dev_df[column].apply(
                    lambda x: x.replace("\n", "<br>")
                )

            hover_template = "<b> prompt:</b> %{customdata[0]}"

        elif NB_LAYERS == 2:
            layers_columns = [c for c in dev_df.columns if c.startswith("layer")]
            for column in layers_columns:
                # Wrap text for display in hover
                dev_df[column] = dev_df[column].apply(
                    lambda x: x.replace("\n", "<br>")
                )

            hover_template = (
                "<b>Layer 1 prompt:</b> %{customdata[0]}"
                "<br><b>Layer 2 prompt:</b> %{customdata[1]}"
            )
    else:
        if NB_LAYERS == 1:
            layers_columns = [
                c for c in dev_df.columns if c.startswith("layer")
            ] + ["input", "output", "label"]
            for column in layers_columns:
                # Wrap text for display in hover
                dev_df[column] = dev_df[column].apply(
                    lambda x: x.replace("\n", "<br>")
                )

            hover_template = (
                "<b>Input:</b> %{customdata[1]}"
                "<br><b>Layer 1 prompt:</b> %{customdata[0]}"
                "<br><b>Output:</b> %{customdata[2]}"
                "<br><b>Label:</b> %{customdata[3]}"
            )

        elif NB_LAYERS == 2:
            layers_columns = [
                c for c in dev_df.columns if c.startswith("layer")
            ] + ["input", "hidden", "output", "label"]
            for column in layers_columns:
                # Wrap text for display in hover
                dev_df[column] = dev_df[column].apply(
                    lambda x: x.replace("\n", "<br>")
                )

            hover_template = (
                "<b>Input:</b> %{customdata[2]}"
                "<br><b>Layer 1 prompt:</b> %{customdata[0]}"
                "<br><b>Hidden:</b> %{customdata[3]}"
                "<br><b>Layer 2 prompt:</b> %{customdata[1]}"
                "<br><b>Output:</b> %{customdata[4]}"
                "<br><b>Label:</b> %{customdata[5]}"
            )
        else:
            raise NotImplementedError()

    hover_config = {
        "customdata": dev_df[layers_columns],
        "hovertemplate": hover_template,
    }

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Make Figure taller
    fig.update_layout(
        autosize=False,
        width=1900,
        height=1000,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    # Add traces
    # text = ["Acc"] * len(df["step"])
    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["run_acc"], name="Running acc", hoverinfo="none"
        ),  # , text=text, **hover_config
        secondary_y=False,
    )

    # text = ["Dev Acc"] * len(df["step"][df["dev_acc"] >= 0])
    fig.add_trace(
        go.Scatter(
            x=dev_df["step"], y=dev_df["dev_acc"], name="Dev acc", **hover_config
        ),
        secondary_y=False,
    )

    # text = ["ELBO"] * len(df["step"])
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["run_elbo"],
            name="Running ELBO",
            hoverinfo="none",
            visible="legendonly",
        ),  # , text=text, **hover_config),
        secondary_y=True,
    )

    # Set x-axes title
    fig.update_xaxes(title_text="steps", nticks=20)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>ELBO</b>", secondary_y=True)

    # Define hover text font and color.
    fig.update_layout(hovermode="x")
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.75)",
            font_size=16,
            font_family="Rockwell",
        ),
    )
    return fig


def main(args):
    datasets = load_dataset_names(args.logfile) if args.logfile else DATASETS
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; margin-bottom: 80px'>Deep Language Networks</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        dataset_selectbox = st.selectbox("Dataset", datasets, index=2, format_func=lambda x: x[1])
        dataset_selectbox = dataset_selectbox[0]
        my_df, candidates, examples = load_data(
            args.logfile or "data.json", dataset_selectbox
        )
        df = my_df

        highlight_example = st.selectbox("Example", [i for i in range(20)], format_func=lambda x: x + 1)
        highlight_step = st.slider("Step", 1, 20)

        st.write("")

        st.markdown(f"**Input:** {examples[examples['step'] == highlight_step]['input'].iloc[highlight_example]}")
        st.markdown(f"**Layer 1 prompt:** {df[df['step'] == highlight_step]['layer_1'].values[0]}")
        if 'layer_2' in df.columns:
            st.markdown(f"**Hidden:** {examples[examples['step'] == highlight_step]['hidden'].iloc[highlight_example]}")
            st.markdown(f"**Layer 2 prompt:** {df[df['step'] == highlight_step]['layer_2'].values[0]}")
        st.markdown(f"**Output:** {examples[examples['step'] == highlight_step]['output'].iloc[highlight_example]}")
        st.markdown(f"**Label:** {examples[examples['step'] == highlight_step]['label'].iloc[highlight_example]}")

    with col2:
        melted_df = df.melt(id_vars=['step'], value_vars=['acc', 'run_acc'], var_name='metric', value_name='value')  
  
        # Create a line chart  
        combined_chart = alt.Chart(melted_df).mark_line().encode(  
            y=alt.Y('value:Q', scale=alt.Scale(domain=[0.4, 1.0])),  
            x='step:Q',  
            color=alt.Color('metric:N', scale=alt.Scale(domain=['acc', 'run_acc'], range=['steelblue', 'lightblue']), legend=alt.Legend(title="Accuracy")),  
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
        st.dataframe(
            candidates[candidates["step"] == highlight_step][
                ["layer_1_candidate", "layer_1_score", "layer_2_candidate", "layer_2_score"]
            ],
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", nargs="?", help="Log file to use (JSON).")
    args = parser.parse_args()

    main(args)
