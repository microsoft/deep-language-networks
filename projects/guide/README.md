# GUIDE - Guided Meta-Prompt Search with Human Feedback

GUIDE is a prototype tool developed as an illustrative use-case for DLN and designed to support researchers and practitioners who wish to explore different ways to reformulate their meta-prompts to guide LLMs’ behavior.  It does so by providing an intuitive interface that collects and tries to integrate user feedback to help refine and direct the search process for alternative meta-prompts. To search for alternative meta-prompts, GUIDE relies on DLN for prompt optimization.


## Limitations and Risks of Using GUIDE

While GUIDE is just a research prototype to show-case how DLN can be used, we believe it is important to acknowledge the limitations and potential risks associated with the use of GUIDE in its current form.

 While GUIDE provides an illustration of a process for exploring different meta-prompt designs, we have not evaluated whether this process yields better meta-prompts given some end user goal. While for the current prototype we have explored a few feedback mechanisms, we have also not evaluated their impact on the resulting meta-prompts or on how the meta-prompt impacts the model behavior at inference time.  Users of GUIDE will thus need to devise their own meta-prompt evaluations.

Given that GUIDE relies on LLMs to generate alternative meta-prompt formulations, the many concerns existing literature has raised about LLMs will hold for GUIDE as well. These include concerns related to generating statements making inaccurate or misleading claims, to surfacing harmful biases and stereotypes, or to generating violent speech, among others.

Finally, given that in practice users are likely to only provide a small number of input examples, GUIDE might end up overfitting on the input examples when generating alternative meta-prompts.  We have not evaluated the impact that the number or type of examples might have on optimization process or the resulting meta-prompt suggestions.


## Getting Started

### Installation

1. Setup a virtual environment using conda or venv
2. Install the requirements using `pip install -r requirements.txt`

### Set your OpenAI API key

Export your key or put it in your *shrc, e.g.: `export OPENAI_API_KEY='...your...key...'`

Please refer to the [DLN main page](../../README.md#set-your-openai-api-key) for instructions on configuring your OpenAI API key for Azure endpoints.


### Usage

Start streamlit app using `streamlit run app.py`


## Serve with Docker

Build the docker image using:

```
docker build -t guide .
```

Run the docker image making sure to pass in your OpenAI API information:

```
docker run --name guide \
    --restart unless-stopped \
    -d \
    -p 8001:8501 \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e OPENAI_API_TYPE=$OPENAI_API_TYPE \
    -e OPENAI_API_BASE=$OPENAI_API_BASE \
    -e OPENAI_API_VERSION=$OPENAI_API_VERSION \
    guide
```
