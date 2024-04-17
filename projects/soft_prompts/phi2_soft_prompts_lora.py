# %% [markdown]
# ## LoRA using Phi-2

# %%
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

from accelerate import Accelerator
accelerator = Accelerator()

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# %%
from peft import PeftModel

model = None
saved_model = None

try:
    sentences = ["Read the following sentence, then determine whether you return to the starting point.\n\nIf you follow these instructions, do you return to the starting point? Take 9 steps. Take 9 steps. Take 4 steps. Turn right.\nOptions:\n- Yes\n- No\n\nAnswer:\n"]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    generate_ids = model.generate(**inputs, max_length=500)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(outputs[0])

    print("Using saved model from data/models/" + model_id)
    saved_model = PeftModel.from_pretrained(model, "data/models/" + model_id + "/lora")
    saved_model.to(device)
    generate_ids = saved_model.generate(**inputs, max_length=500)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(outputs[0])
except ValueError:
    print("Model not found, training new model")

# %%
def preprocess_function(examples, tokenizer, prefix, text_column, label_column, max_length):
    batch_size = len(examples[text_column])
    inputs = [f"{prefix}{x}\n\nAnswer:\n" for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding='max_length', truncation=True, max_length=max_length)

    # Replace padding tokens in the labels with -100
    labels["input_ids"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
def logprobs_for_classes(output_logits, classes):
    logits = [0 for _ in range(len(classes))]
    for i, target in enumerate(classes):
        expanded_classes = [target] + [f" {target}"] + [f"{target.lower()}"] + [f" {target.lower()}"]
        encoded_classes = [tokenizer.encode(c, return_tensors="pt", padding=True).to(device) for c in expanded_classes]
        for token in encoded_classes:
            logits[i] += output_logits[token]
    return F.log_softmax(torch.tensor(logits), dim=0)

# %%
def exact_match_loss(outputs, labels):     
    target_texts = [tokenizer.decode([tok for tok in target if tok != -100], skip_special_tokens=True) for target in labels]
    targets = list(set(target_texts))
    generated_texts = [targets[np.argmax(logprobs_for_classes(out[-1], targets))] for out in outputs.logits]        

    losses = []
    for generated_text, target_text in zip(generated_texts, target_texts):
        generated_tokens = generated_text.split()
        target_tokens = target_text.split()
        loss = sum(generated_token != target_token for generated_token, target_token in zip(generated_tokens, target_tokens))
        losses.append(loss)

    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    total_loss = torch.mean(loss_tensor)
    return total_loss, generated_texts

# %%
def test(dataloader, model, tokenizer, device, exact_match=True):
    total_loss = 0
    test_preds = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss, preds = exact_match_loss(outputs, batch["labels"]) if exact_match else (outputs.loss, [])
        total_loss += loss.detach().float()
        labels = torch.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)
        test_preds.extend(preds)

    total_loss = total_loss / len(dataloader)
    return total_loss, test_preds

# %%
import os
from dln.dataset import init_dataset
from datasets import Dataset, DatasetDict

def load_dln_dataset_to_hf_dataset(dataset_id):
    """Some gynmastics to load the dln dataset into a HuggingFace Dataset.
    dln.dataset should implement an interface compatible with HuggingFace"""

    dln_dataset = init_dataset(
        dataset_id=dataset_id,
        seed=42,
        data_dir=os.path.dirname(os.getcwd()) + "/../data",
    )

    def load_split(split):
        text_data, label_data = dln_dataset.get_data(split)
        data_dict = {"text": text_data, "label": label_data}
        dataset = Dataset.from_dict(data_dict, split=split)
        return dataset

    # Combine the datasets into a DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": load_split("train"),
            "dev": load_split("dev"),
            "test": load_split("test"),
        }
    )
    return dataset_dict

# %%
accelerator = Accelerator()
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Subset

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = "microsoft/phi-2"
tokenizer_name_or_path = "microsoft/phi-2"

dataset_id = "navigate"
initial_instruction = (
    "Read the following question, then choose the correct answer."
)
text_column = "text"
label_column = "label"
max_length = 128
lr = 1e-4
num_epochs = 10
batch_size = 8

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM
)

dataset = load_dln_dataset_to_hf_dataset(dataset_id)

classes = list(set(dataset["train"]["label"]))

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, device_map="auto", padding_side='left')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max(
    [len(tokenizer(class_label)["input_ids"]) for class_label in classes]
)
print(target_max_length)

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
    fn_kwargs={
        "tokenizer": tokenizer,
        "prefix": '',
        "text_column": text_column,
        "label_column": label_column,
        "max_length": max_length,
    },
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["dev"]
test_dataset = processed_datasets["test"]

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=batch_size,
    pin_memory=True,
)
eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=default_data_collator,
    batch_size=batch_size,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    collate_fn=default_data_collator,
    batch_size=batch_size,
    pin_memory=True,
)

if saved_model is None:
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = model.config.eos_token_id
    model = get_peft_model(model, peft_config)
else:
    model = saved_model
    model.enable_input_require_grads()
    print("Using saved model from data/models/" + model_name_or_path)
    
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)

# Send everything through `accelerator.prepare`
train_dataloader, eval_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, test_dataloader, model, optimizer
)

model.eval()
init_test_loss, test_preds = test(test_dataloader, model, tokenizer, device)
init_test_ppl = torch.exp(init_test_loss)  # Perplexity
print(f"Test before training: {init_test_ppl=} {init_test_loss=}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)

        loss = output.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    eval_epoch_loss, eval_preds = test(eval_dataloader, model, tokenizer, device, False)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(torch.tensor(train_epoch_loss))
    print(
        f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
    )

model.eval()
if not saved_model:
    model.save_pretrained("data/models/" + model_name_or_path + "/lora")

final_test_loss, test_preds = test(test_dataloader, model, tokenizer, device)
final_test_ppl = torch.exp(final_test_loss)
print(f"Test before training: {init_test_ppl=} {init_test_loss=}")
print(f"Test after training: {final_test_ppl=} {final_test_loss=}")

# %%
correct = 0
total = 0
for pred, label in zip(test_preds,  dataset['test']['label']):
    if pred.strip() == label.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100

print(f"{accuracy=}% on the test dataset")
print(f"{test_preds[:10]=}")
print(f"{dataset['test']['label'][:10]=}")

"accuracy=84.8% on the test dataset"
"test_preds[:10]=['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']"
"dataset['test']['label'][:10]=['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No']"


