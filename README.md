# Deep Language Networks
<div align="center">

[[ArXiv]](https://arxiv.org/abs/2306.12509)
[[Blog]](https://medium.com/@friederike.niedtner/deep-language-networks-stacking-llms-in-trainable-layers-e7f719bcabde)

</div>

In this repository, you will find the code for
"Deep Language Networks: Joint Prompt Training of Stacked LLMs using Variational Inference".
Please refer to our paper for further details.

## Abstract
We view large language models (LLMs) as stochastic language layers in a network, where the learnable parameters are the natural language prompts at each layer. We stack two such layers, feeding the output of one layer to the next. We call the stacked architecture a Deep Language Network (DLN). We first show how to effectively perform prompt optimization for a 1-Layer language network (DLN-1). We then show how to train 2-layer DLNs (DLN-2), where two prompts must be learnt. We consider the output of the first layer as a latent variable to marginalize, and devise a variational inference algorithm for joint prompt training. A DLN-2 reaches higher performance than a single layer, sometimes comparable to few-shot GPT-4 even when each LLM in the network is smaller and less powerful.

## Setup

### Clone repo
    git clone https://github.com/microsoft/deep-language-networks.git
    cd deep-language-networks

### Installl dependencies
    conda create -n dln python=3.10
    conda activate dln
    pip install -e .

### Setup data
    bash scripts/setup_data.sh

### Set your OpenAI API key

Export your key or put it in your *shrc, e.g.,

    export OPENAI_API_KEY='...your...key...'

In order to use Microsoft Azure endpoints, in addition to the OPENAI_API_KEY,
you need to set the OPENAI_API_TYPE, OPENAI_BASE_URL and OPENAI_API_VERSION.
The OPENAI_API_TYPE must be set to 'azure' and the others correspond to the properties of your endpoint.


> :warning: **Warning:** Setting `echo` and `logprobs` simultaneously is no longer supported for certain OpenAI models.
However, optimizing prompts jointly for 2-DLN using variational inference requires both settings.
To run 2-DLN experiments, consider hosting your own model (see [self-hosted models](#setup-self-hosted-models-vllm)).
Alternatively, you can run 1-DNL by setting output_scoring_function="accuracy" and --one_layer=True.


### Setup self-hosted models (vLLM)

DLN does not directly serve models, instead, we use [vLLM](https://github.com/vllm-project/vllm) to provide an OpenAI-compatible server solution for self-hosted models.

For instructions on setting up an OpenAI-compatible server using vLLM, please follow this [vLLM guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server).

Once your vLLM server is up and running, if you need to load a model from weights on your local machine, please define the path to the tokenizer (e.g., `/path/to/Llama-2-70b-chat-hf`) in the environment variable `TOKENIZER_PATH` so that DLN can load the tokenizer.

Then, set the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables to point to your vLLM server. Finally, remember to unset `OPENAI_API_TYPE` and `OPENAI_API_VERSION` if they were previously set.

    export TOKENIZER_PATH=<PATH_TO_TOKENIZER>  # /path/to/Llama-2-70b-chat-hf
    export OPENAI_API_KEY=<API_KEY>            # EMPTY
    export OPENAI_BASE_URL=<BASE_URL>          # http://127.0.0.1:8000/v1
    unset OPENAI_API_TYPE
    unset OPENAI_API_VERSION


## Datasets

We provide an interface to a few datasets from Big-bench Hard, Leopard, Ordered Prompt, and GSM8K that can be used to train and evaluate DLNs.
See [dln/dataset.py](dln/dataset.py) for more details.

```python
from dln.dataset import init_dataset

dataset = "navigate"
seed = 42
data_dir = "data"

dataset = init_dataset(
    dataset_id=dataset,
    seed=seed,
    data_dir=data_dir,
    n_few_shots=5,
    # max_train_size=max_train_size,
    # max_dev_size=max_dev_size,
    # max_test_size=max_test_size,
)
# Get dataset sizes
dataset.train_size, dataset.dev_size, dataset.test_size

# Get a batch
sentences, labels, few_shot_examples = dataset.get_batch("train", 10)

# Reset the split pointer
dataset.reset_pointer("train")

# Iterate over a dataset split
for sentences, labels, few_shot_examples in dataset.iterate("dev", batch_size=10):
    pass

# Get all data from a split
test_sentences, test_labels = dataset.get_data("test")
```

## LLMs

The easiest way to use LLMs in DLN is to register them using `LLMRegistry`.
You can register different models, or the same model with different configurations.

Any number of keyword arguments can be provided to the `register` method and these will be passed to the model's `generate` method.
Extra keyword arguments can also be provided to the `generate` method, overriding the ones used during the models' instantiation.

```python
from dln.operator import LLMRegistry

llm_registry = LLMRegistry()

fwd_model = llm_registry.register(
    "fwd_model",  # how you refer to the model
    "gpt-35-turbo-instruct",  # model id
    temperature=0.0,
    max_tokens=256,
    stop=None,
)

bwd_model = llm_registry.register(
    "bwd_model",
    "gpt-35-turbo-instruct",
    temperature=0.7,
    max_tokens=512,
    stop=None,
)

fwd_model.generate("What is sirop d'érable?")
fwd_model.generate("What is sirop d'érable?", max_tokens=200, stop=[r"\n"])
```

Alternatively, you can specify the LLMs configuration in a YAML file with the following format:

```yaml
- name: fwd_model  # how you refer to the model
  model: "gpt-35-turbo-instruct"  # model id
  temperature: 0.0  # any generation kwarg
  max_tokens: 256
- name: bwd_model
  model: "gpt-35-turbo-instruct"
  ...
```

This is particularly useful when you want to use models from different APIs. In this case,
you should unset the default `OPENAI` environment vars, and provide them in the YAML file.
For example:

```yaml
- name: phi2-fwd
  model: microsoft/phi-2
  api_key: ${PHI2_API_KEY}
  base_url: ${PHI2_API_BASE}
  api_type: null
  api_version: null
  max_tokens: 256
  temperature: 0.0

- name: gpt-bwd
  model: "gpt-35-turbo-instruct"
  api_key: ${GPT_API_KEY}
  base_url: ${GPT_API_BASE}
  api_type: ${GPT_API_TYPE}
  api_version: ${GPT_API_VERSION}
  temperature: 0.7
  max_tokens: 512

```

Then, you can register the models using the `register_from_yaml` method and get them using the `get` method, as follows:

```python
llm_registry = LLMRegistry.from_yaml("connections.yaml")
fwd_model = llm_registry.get("phi2-fwd")
bwd_model = llm_registry.get("gpt-bwd")

output = bwd_model.generate("Why do programmers prefer dark mode?")

# You can always provide extra keyword arguments to the `generate` method,
# which will override the ones provided when instantiating the models.

output = bwd_model.generate(
    "Why do programmers prefer dark mode?",
    max_tokens=100,
    echo=True,
)
```

## Losses, Samplers and Scores

DLN provides a few losses that can be found in [dln/losses.py](dln/loss.py). A simple example of how to use them is as follows:

```python
from dln.loss import LossRegistry
from dln.postprocessing import postprocess_prediction

LossRegistry.available_losses()  # list available losses
loss_fn = LossRegistry.instantiate(
    "exact_match_loss", postprocess_prediction
)
y = ["Montreal", "Toronto", "Sao Paulo"]
y_hat = ["Montréal", "Toronto", "SaoPaulo"]
losses = loss_fn(y_hat, y)
# array([1., 0., 1.], dtype=float32)
```
For sampling and scoring both prompts and hidden states for the Variational Inference algorithm, samplers are found in [dln/vi/sampler.py](dln/vi/sampler.py), and the LogProbsScore in [dln/score.py](dln/score.py). Samplers use templates that are found in [dln/templates.py](dln/templates.py).


```python
import numpy as np
from dln.operator import LLMRegistry
from dln.vi.sampler import PosteriorSampler, PromptSampler
from dln.score import LogProbsScore, ScoreRequest

llm_registry = LLMRegistry()
llm = llm_registry.register(
    "llm",
    "microsoft/phi-2",
)

prompt_sampler = PromptSampler(llm, "q_action_prompt")
posterior_sampler = PosteriorSampler(llm, "suffix_forward_tbs")
logprobs_score = LogProbsScore(llm)

prompt_proposals = prompt_sampler.sample_q_p(
    inputs=["France", "Canada", "Brazil"],
    y=["Paris", "Ottawa", "Brasilia"],
    y_hat=["Paris", "Ottawa", "Sao Paulo"],
    losses=[0, 0, 0, 1],
    prompt="What is the capital of this country",
    num_samples=10,
)  # sample prompts

hidden_states = posterior_sampler.sample_q_h(
    x=np.array(["France", "Canada", "Brazil"]),
    y=["Paris", "Ottawa", "Brasilia"],
    h=["Paris", "Toronto", "Sao Paulo"],
    prompt="What is the largest city in this country",
    next_prompt="What is the capital of this country",
    num_samples=10,
)

score_request = ScoreRequest(
    context="What is the capital of this country: Canada",
    target="Ottawa",
    payload="Ottawa",
)
score = logprobs_score.score_requests([score_request])
# LogProbs(logp_targets=array([-7.67090403]), distribution=array([-3.02606859]))
```


You can refer to [vi_main.py](projects/vi_dln/vi_main.py) for a complete example of how to use the DLN components.


## Variational Inference experiments

Please see the [Variational Inference README](projects/vi_dln/README.md) for information on how to run VI experiments.


## Limitations

When it comes to large-scale natural language models, there are particular fairness and responsible AI issues to consider.
People use language to describe the world and to express their beliefs, assumptions, attitudes, and values.
As a result, publicly available text data typically used to train large-scale natural language models contains
societal biases relating to race, gender, religion, age, and other groups of people, as well as other undesirable content.
These societal biases are reflected in the distributions of words, phrases, and syntactic structures.
Large-scale natural language models trained with such data can potentially behave in ways that are unfair,
unreliable, or offensive, in turn causing harms.

While we are fully aware of the limitations of addressing societal issues through technical work,
we hope that modular approaches like ours will alleviate some of the issues associated with LLMs,
like the concentration of power associated with the difficulty to train them. We also hope that,
by facilitating the reusability and adaptivity of such models, we shall make them more amenable to a wider variety of use cases.
However, while we discuss the performance of these models on artificial benchmarks,
we do not address the question of when and how such models should be deployed,
nor do we offer additional guarantees against their misuse. We also emphasize that performance on artificial tasks,
even if realistic, is neither representative of performance in uncontrolled environments,
nor enough to justify the deployment of these models in high stakes situations.
Please refer to our paper for the specific evaluations we conducted.

## Citing Deep Language Networks
If you find DLNs useful, please consider citing this work!

```text
@article{sordoni2023deep,
      title={Deep Language Networks: Joint Prompt Training of Stacked LLMs using Variational Inference},
      author={Alessandro Sordoni and Xingdi Yuan and Marc-Alexandre Côté and Matheus Pereira and Adam Trischler and Ziang Xiao and Arian Hosseini and Friederike Niedtner and Nicolas Le Roux},
      year={2023},
      eprint={2306.12509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
