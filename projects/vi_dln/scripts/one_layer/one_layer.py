import json
import subprocess

import click


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Dataset name",
    required=True,
)
def main(dataset):
    config_file="one-layer-dln-hp-search-result.json"
    with open(config_file) as f:
        config_data = json.load(f)

    config = config_data[dataset]
    q_prompt_tpl = config["hyperparam"]["q_prompt_tpl"]
    tolerance = config["hyperparam"]["tolerance"]
    use_memory = config["hyperparam"]["use_memory"]
    held_out_prompt_ranking = config["hyperparam"]["held_out_prompt_ranking"]

    output_dir = f"log/one_layer/{dataset}"
    for seed in [13, 42, 25]:
        command = list(map(str, [
            "python",
            "vi_main.py",
            "--balance_batch",
            "--num_p_samples", 20,
            "--bwd_temp", 0.7,
            "--iters", 20,
            "--p_class", "classify_forward:3.0",
            "--q_prompt", q_prompt_tpl,
            "--out_dir", output_dir,
            "--batch_size", 20,
            "--seed", seed,
            "--dataset", dataset,
            "--tolerance", tolerance,
            "--use_memory", use_memory,
            "--held_out_prompt_ranking", held_out_prompt_ranking,
            "--one_layer", True,
            "--do_first_eval",
        ]))
        print(' '.join(command))
        subprocess.run(command)

if __name__ == "__main__":
    main()