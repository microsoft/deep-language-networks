# Deep Language Networks

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
you need to set the OPENAI_API_TYPE, OPENAI_API_BASE and OPENAI_API_VERSION.
The OPENAI_API_TYPE must be set to 'azure' and the others correspond to the properties of your endpoint.


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
