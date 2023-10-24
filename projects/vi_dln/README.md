# Variational Inference

## Setup

Please follow the instructions from the [main README](../../README.md).


## Reproducing results


### One-Layer DLN

See [one-layer-dln-hp-search-result.json](./one-layer-dln-hp-search-result.json) for available datasets and the hyperparameter search results. Then run:

    python scripts/one_layer/one_layer.py --dataset <dataset_id>


### Two-Layer DLN

For two-layer DLN, you can select one of the following training strategies:

- `two_layers_fix_2nd`: Load pretrained prompts from [one-layer-dln-hp-search-result.json](./one-layer-dln-hp-search-result.json) to the second layer and train only the first layer.

- `two_layers_ft_2nd`: Load pretrained prompts from [one-layer-dln-hp-search-result.json](./one-layer-dln-hp-search-result.json) to the second layer and train both the first and second layers.

- `two_layers_e2e`: Train the two layers from scratch.

Then, run the following command:

    bash scripts/<training_strategy>/<dataset_id>.sh

Results will be saved in `log/<training_strategy>/<dataset_id>`.


## Running your own experiments

If you decide to run your own experiments, please check:

    python vi_main.py --help


## LLMs yaml config

You can specify multiple LLMs for the same execution using yaml, for example:

```yaml
- name: gpt-bwd
  model: text-davinci-003
  api_key: OPENAI_API_KEY
  api_base: OPENAI_API_BASE
  api_type: OPENAI_API_TYPE
  api_version: OPENAI_API_VERSION
  temperature: 0.7
  max_tokens: 1024

- name: llama2-fwd
  model: meta-llama/Llama-2-70b-chat-hf
  api_key: EMPTY
  api_base: http://127.0.0.1:8000/v1
  api_type: null
  api_version: null
  max_tokens: 512
  temperature: 0

- name: llama2-posterior
  model: meta-llama/Llama-2-70b-chat-hf
  api_key: EMPTY
  api_base: http://127.0.0.1:8000/v1
  api_type: null
  api_version: null
  max_tokens: 512
  temperature: 0.75
```