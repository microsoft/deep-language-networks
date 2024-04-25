FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
WORKDIR /app

RUN git clone -b deep-language-networks-soft-prompts https://github.com/microsoft/deep-language-networks
WORKDIR /app/deep-language-networks

RUN pip install -e .
RUN bash scripts/setup_data.sh