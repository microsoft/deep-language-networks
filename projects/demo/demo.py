import argparse
import json
import textwrap

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from jinja2 import Template
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


def main(args):
    app = dash.Dash()
    app.layout = html.Div(
        [
            html.H2(
                "Deep Language Networks",
                style={
                    "textAlign": "center",
                },
            ),
            dcc.Dropdown(
                id="dataset_dropdown",
                options=[
                    {"label": f"{title}", "value": id_} for id_, title in DATASETS
                ],
                value="subj",
                multi=False,
                style={
                    "backgroundColor": "rgb(229, 236, 246)",
                    "margin": "10px 0",
                },
            ),
            dcc.Dropdown(
                id="example_dropdown",
                options=[
                    {
                        "label": f"Example {i}" if i > 0 else "Show only prompts",
                        "value": i,
                    }
                    for i in range(0, 20 + 1)
                ],
                value=0,  # df['id'].iloc[0],
                multi=False,
                style={
                    "backgroundColor": "rgb(229, 236, 246)",
                    "margin": "10px 0",
                },
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div(
                id="table-container",
                style={
                    "backgroundColor": "rgb(229, 236, 246)",
                    "margin": "10px 0",
                    "padding": "10px",
                },
            ),
        ]
    )

    # Create a callback to update the table-container
    @app.callback(
        Output("table-container", "children"),
        [Input("scatter-plot", "hoverData"), Input("dataset_dropdown", "value")],
    )  # coulbe be either clickData, hoverData
    def update_table(callbackData, dataset_dropdown):
        df, candidates, examples = load_data(
            args.logfile or "data.json", dataset_dropdown
        )

        # Merge layers and examples
        df = df.merge(examples, on="step", how="left")

        step = callbackData["points"][0]["x"] if callbackData is not None else 0
        filtered_df = candidates[candidates["step"] == step]
        table = dbc.Table.from_dataframe(
            filtered_df, striped=True, bordered=True, hover=True
        )
        return table

    @app.callback(
        Output("scatter-plot", "figure"),
        [Input("example_dropdown", "value"), Input("dataset_dropdown", "value")],
    )
    def update_scatter_plot(example_dropdown, dataset_dropdown):
        df, candidates, examples = load_data(
            args.logfile or "data.json", dataset_dropdown
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

    app.run_server(debug=args.debug, host=args.dash_host or "127.0.0.1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", nargs="?", help="Log file to use (JSON).")
    parser.add_argument(
        "--dash-host", help="Host for Dash (setting this, implies --dash)."
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="Launch in debug mode."
    )
    args = parser.parse_args()

    main(args)
