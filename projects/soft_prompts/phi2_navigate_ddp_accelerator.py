import os
import torch
import numpy as np
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.utils.data import Subset
from dln.dataset import init_dataset
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    PeftConfig,
    PeftModel,
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
from accelerate import Accelerator
accelerator = Accelerator()

def load_dln_dataset_to_hf_dataset(dataset_id):
    """Some gynmastics to load the dln dataset into a HuggingFace Dataset.
    dln.dataset should implement an interface compatible with HuggingFace"""

    dln_dataset = init_dataset(
        dataset_id=dataset_id,
        seed=42,
        data_dir=os.path.dirname(os.path.realpath(__file__)) + "/../../data",
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

def logprobs_for_classes(tokenizer, output_logits, classes):
    logits = [0 for _ in range(len(classes))]
    for i, target in enumerate(classes):
        expanded_classes = [target] + [f" {target}"] + [f"{target.lower()}"] + [f" {target.lower()}"]
        encoded_classes = [tokenizer.encode(c, return_tensors="pt", padding=True) for c in expanded_classes]
        for token in encoded_classes:
            logits[i] += output_logits[token]
    return nn.functional.log_softmax(torch.tensor(logits), dim=0)

def exact_match_loss(tokenizer, outputs, labels):     
    target_texts = [tokenizer.decode([tok for tok in target if tok != -100], skip_special_tokens=True) for target in labels]
    targets = list(set(target_texts))
    generated_texts = [targets[np.argmax(logprobs_for_classes(tokenizer, out[-1], targets))] for out in outputs.logits]        

    losses = []
    for generated_text, target_text in zip(generated_texts, target_texts):
        generated_tokens = generated_text.split()
        target_tokens = target_text.split()
        loss = sum(generated_token != target_token for generated_token, target_token in zip(generated_tokens, target_tokens))
        losses.append(loss)

    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    total_loss = torch.mean(loss_tensor)
    return total_loss

def test(dataloader, model, tokenizer, device, exact_match=True):
    total_loss = 0
    preds = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = exact_match_loss(tokenizer, outputs, batch["labels"]) if exact_match else outputs.loss
        total_loss += loss.detach().float()
        labels = torch.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)

    total_loss = total_loss / len(dataloader)
    return total_loss

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


def main():
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
    lr = 3e-2
    num_epochs = 10
    batch_size = 8

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text=initial_instruction,
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
    try:
        saved_model = PeftModel.from_pretrained(model, os.path.dirname(os.path.realpath(__file__)) + "/data/models/" + model_name_or_path)
        model = saved_model
        print("Using saved model from data/models/" + model_name_or_path)
    except ValueError:
        model = get_peft_model(model, peft_config)
        print("Model not found, training new model")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model = model.to(device)

    # Send everything through `accelerator.prepare`
    train_loader, eval_loader, test_loader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader, model, optimizer
    )

    model.eval()
    init_test_loss = test(test_dataloader, model, tokenizer, device)
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
        eval_epoch_loss = test(eval_dataloader, model, tokenizer, device, False)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(train_epoch_loss))
        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )

    model.eval()
    if not saved_model:
        model.save_pretrained("data/models/" + model_name_or_path)

    final_test_loss = test(test_dataloader, model, tokenizer, device)
    final_test_ppl = torch.exp(final_test_loss)
    print(f"Test before training: {init_test_ppl=} {init_test_loss=}")
    print(f"Test after training: {final_test_ppl=} {final_test_loss=}")

if __name__ == "__main__":
    main()

# accelerate launch phi2_navigate_ddp_accelerator.py

# The following values were not passed to `accelerate launch` and had defaults used instead:
#         `--num_processes` was set to a value of `4`
#                 More than one GPU was found, enabling multi-GPU training.
#                 If this was unintended please pass in `--num_processes=1`.
#         `--num_machines` was set to a value of `1`
#         `--mixed_precision` was set to a value of `'no'`
#         `--dynamo_backend` was set to a value of `'no'`
