# Deep Language Networks


## Setup

### Clone repo
    git clone https://github.com/xingdi-eric-yuan/deep_language_networks.git
    cd deep_language_networks

### Installl dependencies
    conda create -n nll python=3.10
    conda activate nll
    pip install -e .

### Setup data
    bash scripts/setup_data.sh

### Set your OpenAI API key

Export your key or put it in your *shrc, e.g.,

    export OPENAI_API_KEY='...your...key...'

In order to use Microsoft Azure endpoints, in addition to the OPENAI_API_KEY,
you need to set the OPENAI_API_TYPE, OPENAI_API_BASE and OPENAI_API_VERSION.
The OPENAI_API_TYPE must be set to 'azure' and the others correspond to the properties of your endpoint.


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
