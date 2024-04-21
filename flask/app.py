# a simple Flask API to emulate OpenAI's using llama models and/or transformers
# runs on 3080

import sys
import time
import torch
import json
from peft import PeftModel

from flask import Flask, make_response, request, abort
from flask.json import jsonify

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import scan_cache_dir

from transformers import AutoModelForCausalLM, AutoTokenizer

# tested on a 3080
LOAD_8BIT = False
BASE_MODEL = "gpt2"

# clues from :
# https://github.com/shawwn/openai-server
# https://github.com/jquesnelle/transformers-openai-api
# https://github.com/facebookresearch/metaseq
# https://github.com/tloen/alpaca-lora

# requirement: pip3 install transformers huggingface_hub flask
# requirement: pip3 install sentencepiece
# requirement: pip3 install git+https://github.com/huggingface/transformers.git
# requirement: pip3 install accelerate
# requirement: pip3 install bitsandbytes
# requirement: pip3 install git+https://github.com/huggingface/peft.git
# requirement: pip3 install loralib

# set up the Flask application
app = Flask(__name__)

cached_model=""
tokenizer=None
model=None
models = {}

llamaModels =[ 
              'gpt2',
              'microsoft/phi-2',
              'llama-7b-hf',
              'alpaca-7b-hf',
              'decapoda-research/llama-7b-hf',
              'tloen/alpaca-lora-7b', 
              'decapoda-research/llama-7b-hf-int4', 
              'decapoda-research/llama-13b-hf-int4',  
              'decapoda-research/llama-65b-hf-int4',  
              'decapoda-research/llama-30b-hf-int4',  
              'decapoda-research/llama-30b-hf',  
              'decapoda-research/llama-65b-hf',  
              'decapoda-research/llama-13b-hf',  
              'decapoda-research/llama-smallint-pt',
              'decapoda-research/llama-7b-hf-int8',                  
            ]

# collect the models available in the cache
report = scan_cache_dir()

modelList = []
for repo in report.repos:

    print("repo_id:",json.dumps(repo.repo_id,indent=4))
    print("repo_type:",json.dumps(repo.repo_type,indent=4))
    print("repo_path:",json.dumps(str(repo.repo_path),indent=4))
    #print("revisions",json.dumps(str(repo.revisions),indent=4))
    print("size_on_disk:",json.dumps(repo.size_on_disk,indent=4))
    print("nb_files:",json.dumps(repo.nb_files,indent=4))
    #print(json.dumps(repo.str(refs),indent=4))
    alias = repo.repo_id
    if ('/' in repo.repo_id):
        alias = repo.repo_id.split('/')[1]

    modelList.append(alias)
    models[alias] = repo.repo_id
    print()

for modelname in llamaModels:
    alias = modelname
    if ('/' in modelname):
        alias = modelname.split('/')[1]
    models[alias] = modelname
    modelList.append(alias)

modelList.sort()


print("Available models:")
for model in modelList:
    print(model)

# find out which device we are using
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

print("Using device: {}".format(device))

#set up the llama model
if device == "cuda":
    lmodel = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
        resume_download=True
    )
    # lmodel = PeftModel.from_pretrained(
    #     lmodel,
    #     torch_dtype=torch.float16,
    # )
   
elif device == "mps":
    lmodel = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
        resume_download=True
    )
    lmodel = PeftModel.from_pretrained(
        lmodel,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    lmodel = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    # lmodel = PeftModel.from_pretrained(
    #     model=lmodel,
    #     device_map={"": device},
    #     resume_download=True
    # )

ltokenizer = AutoTokenizer.from_pretrained("gpt2", resume_download=True)

def generate_prompt_llama(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

# if not LOAD_8BIT:
#     lmodel.half()  # seems to fix bugs for some users.

lmodel.eval()

# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)


def evaluate_llama(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt_llama(instruction, input)
    print(f"prompt: {prompt}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"top_k: {top_k}")
    print(f"num_beams: {num_beams}")
    print(f"max_new_tokens: {max_new_tokens}")
       
    inputs = ltokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     **kwargs,
    # )
    with torch.no_grad():
        generation_output = lmodel.generate(
            input_ids=input_ids,
            # generation_config=generation_config,
            # return_dict_in_generate=True,
            # output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    # s = generation_output.sequences[0]
    output = ltokenizer.decode(generation_output[-1])
    print(f"output: {output}")
    gen_text = output.split("### Response:")[1].strip()
    print(f"gen_text: {gen_text}")
    return gen_text
    #return output.split("### Response:")[1].strip()


def update_model(model_name):
    global cached_model,llamaModels,ltokenizer,lmodel
        # is it an alias?
    if (model_name in models):
        model_name = models[model_name]

    if (model_name in llamaModels) and (model_name != cached_model):
        print("Using llama model: {}".format(model_name))
        tokenizer = ltokenizer
        model = lmodel
        return ltokenizer, lmodel
    
    if model_name != cached_model:
        print("Loading model: {}".format(model_name))
        cached_model = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name,resume_download=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,resume_download=True)
        model.to("cuda")
    return tokenizer, model

def decode_kwargs(data):
    # map the data to the kwargs (openai to huggingface)
    kwargs = {}
    if 'n' in data:
        kwargs['num_return_sequences'] = data['n']
    if 'stop' in data:
        kwargs['early_stopping'] = True
        kwargs['stop_token'] = data['stop']
    if 'suffix' in data:
        kwargs['suffix'] = data['suffix']
    if 'presence_penalty' in data:
        kwargs['presence_penalty'] = data['presence_penalty']
    if 'frequency_penalty' in data:
        kwargs['repetition_penalty'] = data['frequency_penalty']
    if 'repetition_penalty ' in data:
        kwargs['repetition_penalty'] = data['repetition_penalty ']
    if 'best_of ' in data:
        kwargs['num_return_sequences'] = data['best_of ']

    #kwargs['do_sample'] = True
    #for key, value in data.items():
    #    if key in ["temperature", "top_p", "top_k", "num_beams", "max_new_tokens"]:
    #        kwargs[key] = value
    return kwargs

# define the completion endpoint
@app.route("/v1/engines/<model_name>/completions", methods=["POST"])
def completions(model_name):
    # get the request data
    data = request.get_json(force=True)
    # is it an alias?
    if (model_name in models):
        model_name = models[model_name]
   
    #update model
    tokenizer, model = update_model(model_name)

    # get the prompt and other parameters from the request data
    prompt = data["prompt"]

    max_tokens = data.get("max_tokens", 16)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.75)
    top_k = data.get("top_k", 40)
    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    kwargs = decode_kwargs(data)
    # generate the completion
    
    if (model_name in llamaModels):
        #generated_text = evaluate_llama(prompt,**kwargs)
        generated_text = evaluate_llama(prompt,
                                        #input = prompt,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        num_beams=num_beams,
                                        max_new_tokens=max_new_tokens,
                                        **kwargs)

    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids=input_ids,
                                max_length=max_tokens, 
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                **kwargs)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))
    total_tokens = prompt_tokens + completion_tokens
    return jsonify( {
            'object': 'text_completion',
            'id': 'dummy',
            'created': int(time.time()),
            'model': model_name,
            'choices': 
                [{'text': generated_text, 'finish_reason': 'length'}],
            'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                    }
                }
            )

    # return the response data
    # return jsonify(response.choices[0].text)
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    # get the request data
    data = request.get_json(force=True)
    model_name = data["model"]
    messages = data["messages"]
    # generate prompt from messages
    # messages must be an array of message objects, where each object has a role (either "system", "user", or "assistant") and content (the content of the message). 

    prompt = ""
    for message in messages:
        prompt += message["role"] + ": " + message["content"] + "\n"
    #prompt += "assistant: "

    # is it an alias?
    if (model_name in models):
        model_name = models[model_name]
   
    #update model
    tokenizer, model = update_model(model_name)


    # get the prompt and other parameters from the request data
    #prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 16)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.75)
    top_k = data.get("top_k", 40)
    num_beams = data.get("num_beams", 1)
    max_new_tokens = data.get("max_new_tokens", 256)

    kwargs = decode_kwargs(data)

    if (model_name in llamaModels):
        #generated_text = evaluate_llama_chat(prompt,**kwargs)
        instruction = "Be a generallly helpful assistiang chatting with the user. Return the response for the assistant."
        generated_text = evaluate_llama(instruction,
                                        input = prompt,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        num_beams=num_beams,
                                        max_new_tokens=max_new_tokens,
                                        **kwargs)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids=input_ids,
                                max_length=max_tokens, 
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                **kwargs)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))
    total_tokens = prompt_tokens + completion_tokens
    return jsonify( {
            'object': 'text_completion',
            'id': 'dummy',
            'created': int(time.time()),
            'model': model_name,
            'choices': 
                [{'role':'assistant','content': generated_text, 'finish_reason': 'stop'}],
            'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                    }
                }
            )

    # return the response data
    # return jsonify(response.choices[0].text)

@app.route('/v1/completions', methods=['POST'])
def v1_completions():
    print("COMPLETION REQUEST", request.json)
    return completions(request.json['model'])

# define the engines endpoint    
@app.route('/v1/engines')
@app.route('/v1/models')
def v1_engines():
    return make_response(jsonify({
        'data': [{
            'object': 'engine',
            'id': id,
            'ready': True,
            'owner': 'huggingface',
            'permissions': None,
            'created': None
        } for id in models.keys()]
    }))


if __name__ == "__main__":
    app.run()


"""
curl http://127.0.0.1:5000/v1/completions -v -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" --data "{\"model\":\"alpaca-lora-7b\",\"prompt\":\"Say this is a test\",\"max_tokens\":7,\"temperature\":0}"
*   Trying 127.0.0.1:5000...
* Connected to 127.0.0.1 (127.0.0.1) port 5000 (#0)
> POST /v1/completions HTTP/1.1
> Host: 127.0.0.1:5000
> User-Agent: curl/7.83.1
> Accept: */*
> Content-Type: application/json
> Authorization: Bearer $OPENAI_API_KEY
> Content-Length: 87
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: Werkzeug/2.2.3 Python/3.10.9
< Date: Fri, 24 Mar 2023 22:19:13 GMT
< Content-Type: application/json
< Content-Length: 226
< Connection: close
<
{"choices":[{"finish_reason":"length","text":"This is a test."}],"created":1679696353,"id":"dummy","model":"tloen/alpaca-lora-7b","object":"text_completion","usage":{"completion_tokens":6,"prompt_tokens":6,"total_tokens":12}}
* Closing connection 0

curl http://127.0.0.1:5000/v1/chat/completions -v -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" --data "{\"model\":\"alpaca-lora-7b\",\"max_tokens\":64,\"temperature\":0.95,  \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"
*   Trying 127.0.0.1:5000...
* Connected to 127.0.0.1 (127.0.0.1) port 5000 (#0)
> POST /v1/chat/completions HTTP/1.1
> Host: 127.0.0.1:5000
> User-Agent: curl/7.83.1
> Accept: */*
> Content-Type: application/json
> Authorization: Bearer $OPENAI_API_KEY
> Content-Length: 115
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: Werkzeug/2.2.3 Python/3.10.9
< Date: Fri, 24 Mar 2023 22:25:01 GMT
< Content-Type: application/json
< Content-Length: 257
< Connection: close
<
{"choices":[{"content":"Hello! How can I help you?","finish_reason":"stop","role":"assistant"}],"created":1679696701,"id":"dummy","model":"tloen/alpaca-lora-7b","object":"text_completion","usage":{"completion_tokens":9,"prompt_tokens":6,"total_tokens":15}}
* Closing connection 0

curl http://127.0.0.1:5000/v1/models

"""