# %% [markdown]
# ## Multitask prompt tuning using Phi-2

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

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

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

    task_ids = [0 for i in labels["input_ids"]]
    task_ids = torch.tensor(task_ids)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["task_ids"] = task_ids
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
def test(dataloader, model1, model2=None, exact_match=True):
    total_loss = 0
    test_preds = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output1 = model1(**batch, output_hidden_states=True)
            loss, preds = exact_match_loss(output1, batch["labels"]) if exact_match else (output1.loss, [])
        if model2:
            inputs_embeds = output1.hidden_states[-1]
            sequence_length = inputs_embeds.shape[1]
            labels = batch['labels']
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
            padding = torch.full((labels.shape[0], sequence_length - labels.shape[1]), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([padding, labels], dim=1).to(device)
            task_ids = torch.tensor([1 for i in batch["task_ids"]]).to(device)
            output2 = model2(inputs_embeds=inputs_embeds, labels=labels, task_ids=task_ids, attention_mask=attention_mask, output_hidden_states=True)
            loss, preds = exact_match_loss(output2, batch["labels"]) if exact_match else (output2.loss, [])
    
        total_loss += loss.detach().float()
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
        data_dir="../../data",
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
from accelerate import Accelerator
accelerator = Accelerator()

from peft import (
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    TaskType,
    PeftModel,
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

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = "microsoft/phi-2"
tokenizer_name_or_path = "microsoft/phi-2"

dataset_id = "navigate"
initial_instruction = (
    "Read the following sentence, then determine whether you return to the starting point."
)
text_column = "text"
label_column = "label"
max_length = 128
lr = 3e-2
num_epochs = 50
batch_size = 8

peft_config = MultitaskPromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_tasks=2,
    prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text=initial_instruction,
    num_transformer_submodules=1,
    tokenizer_name_or_path=model_name_or_path,
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

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.config.pad_token_id = model.config.eos_token_id
try:
    saved_model1 = PeftModel.from_pretrained(model, "data/models/" + model_name_or_path + "/model1")
    saved_model2 = PeftModel.from_pretrained(model, "data/models/" + model_name_or_path + "/model2")
    model1 = saved_model1
    model2 = saved_model2
    print("Using saved model from data/models/" + model_name_or_path)
except ValueError:
    model1 = get_peft_model(model, peft_config)
    model2 = get_peft_model(model, peft_config)
    print("Model not found, training new model")

optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr)
lr_scheduler1 = get_linear_schedule_with_warmup(
    optimizer=optimizer1,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),

)
lr_scheduler2 = get_linear_schedule_with_warmup(
    optimizer=optimizer2,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model1 = model1.to(device)
model2 = model2.to(device)

# Send everything through `accelerator.prepare`
train_dataloader, eval_dataloader, test_dataloader, model, model1, model2, optimizer1, optimizer2 = accelerator.prepare(
    train_dataloader, eval_dataloader, test_dataloader, model, model1, model2, optimizer1, optimizer2
)

model1.eval()
model2.eval()

init_test_loss1, test_preds1 = test(test_dataloader, model1)
init_test_loss2, test_preds2 = test(test_dataloader, model1, model2)
init_test_ppl1 = torch.exp(init_test_loss1)  # Perplexity
init_test_ppl2 = torch.exp(init_test_loss2)  # Perplexity
print(f"Test before training1: {init_test_ppl1=} {init_test_loss1=}")
print(f"Test before training2: {init_test_ppl2=} {init_test_loss2=}")

for epoch in range(num_epochs):
    model1.train()
    model2.train()
    total_loss1 = 0
    total_loss2 = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output1 = model1(**batch, output_hidden_states=True)
        
        inputs_embeds = output1.hidden_states[-1]
        sequence_length = inputs_embeds.shape[1]
        labels = batch['labels']
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
        padding = torch.full((labels.shape[0], sequence_length - labels.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([padding, labels], dim=1).to(device)
        task_ids = torch.tensor([1 for i in batch["task_ids"]]).to(device)
        output2 = model2(inputs_embeds=inputs_embeds, labels=labels, task_ids=task_ids, attention_mask=attention_mask, output_hidden_states=True)
       
        loss1 = output1.loss
        loss2 = output2.loss
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        accelerator.backward(loss2, retain_graph=True)
        accelerator.backward(loss1)
        optimizer1.step()
        optimizer2.step()
        lr_scheduler1.step()
        lr_scheduler2.step()

    model1.eval()
    model2.eval()
    eval_epoch_loss1, eval_preds1 = test(eval_dataloader, model1, None, False)
    eval_epoch_loss2, eval_preds2 = test(eval_dataloader, model1, model2, False)
    eval_ppl1 = torch.exp(eval_epoch_loss1)
    eval_ppl2 = torch.exp(eval_epoch_loss2)
    train_epoch_loss1 = total_loss1 / len(train_dataloader)
    train_epoch_loss2 = total_loss2 / len(train_dataloader)
    train_ppl1 = torch.exp(torch.tensor(train_epoch_loss1))
    train_ppl2 = torch.exp(torch.tensor(train_epoch_loss2))
    print(
        f"{epoch=}: {train_ppl1=} {train_epoch_loss1=} {eval_ppl1=} {eval_epoch_loss1=}"
    )
    print(
        f"{epoch=}: {train_ppl2=} {train_epoch_loss2=} {eval_ppl2=} {eval_epoch_loss2=}"
    )

model1.eval()
model2.eval()
if not saved_model1:
    model1.save_pretrained("data/models/" + model_name_or_path + "/model1")
if not saved_model2:
    model2.save_pretrained("data/models/" + model_name_or_path + "/model2")

final_test_loss1, test_preds1 = test(test_dataloader, model1)
final_test_loss2, test_preds2 = test(test_dataloader, model1, model2)
final_test_ppl1 = torch.exp(final_test_loss1)
final_test_ppl2 = torch.exp(final_test_loss2)
print(f"Test before training1: {init_test_ppl1=} {init_test_loss1=}")
print(f"Test before training2: {init_test_ppl2=} {init_test_loss2=}")
print(f"Test after training1: {final_test_ppl1=} {final_test_loss1=}")
print(f"Test after training2: {final_test_ppl2=} {final_test_loss2=}")

# %%
correct = 0
total = 0
for pred, label in zip(test_preds1,  dataset['test']['label']):
    if pred.strip() == label.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100

print(f"{accuracy=}% on the test dataset")
print(f"{test_preds1[:10]=}")
print(f"{dataset['test']['label'][:10]=}")

"accuracy=80.4% on the test dataset"
"test_preds[:10]=['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']"
"dataset['test']['label'][:10]=['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No']"

# %%
correct = 0
total = 0
for pred, label in zip(test_preds2,  dataset['test']['label']):
    if pred.strip() == label.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100

print(f"{accuracy=}% on the test dataset")
print(f"{test_preds2[:10]=}")
print(f"{dataset['test']['label'][:10]=}")

"accuracy=82.39999999999999% on the test dataset"
"test_preds[:10]=['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']"
"dataset['test']['label'][:10]=['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No']"

# %%
sentences = ["Read the following sentence, then determine whether you return to the starting point.\n\nIf you follow these instructions, do you return to the starting point? Take 9 steps. Take 9 steps. Take 4 steps. Turn right.\nOptions:\n- Yes\n- No\n\nAnswer:\n"]
inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

task_ids = [1 for i in inputs["input_ids"]]
task_ids = torch.tensor(task_ids).to(device)

generate_ids = model2.generate(**inputs, max_length=100, task_ids=task_ids)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(outputs[0])
["No"]


