# %% [markdown]
# ## Multitask prompt tuning using Phi-2

# %%
import os
import dln
import copy
import torch
import click
import wandb
import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import random

model = None
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
def logprobs_for_classes(model, batch, output_logits, classes):
    logits = [torch.zeros_like(output_logits[0]) for _ in range(len(classes))]
    for i, target in enumerate(classes):
        expanded_classes = [target] + [f" {target}"] + [f"{target.lower()}"] + [f" {target.lower()}"]
        encoded_classes = [tokenizer.encode(c, return_tensors="pt", padding=True).to(device) for c in expanded_classes]
        logits[i] = logits[i].expand_as(output_logits[0])
        input_sequence = copy.deepcopy(batch)
        for token in encoded_classes:
            for _ in range(len(token[0])):
                output_logits = model(**input_sequence, output_hidden_states=True).logits
                logits[i] = logits[i].expand_as(output_logits[0])
                logprobs = F.softmax(output_logits, dim=-1)
                next_token = torch.multinomial(logprobs[-1][-1], num_samples=1)
                logits[i] = logits[i].expand_as(output_logits[0])
                # logits[i] += output_logits[next_token]
                # input_sequence = torch.cat([input_sequence, next_token], dim=-1)
            # for j in range(len(token[0])):
            #     logits[i] += output_logits[token[0][j]]
    return F.log_softmax(torch.stack(logits), dim=0).cpu()

# %%
def exact_match_loss(model, batch, outputs):
    labels = batch["labels"]
    target_texts = [tokenizer.decode([tok for tok in target if tok != -100], skip_special_tokens=True) for target in labels]
    targets = list(set(target_texts))
    generated_texts = [targets[np.argmax(logprobs_for_classes(model, batch, out[-1], targets))] for out in outputs.logits]        

    losses = []
    for generated_text, target_text in zip(generated_texts, target_texts):
        generated_tokens = generated_text.split()
        target_tokens = target_text.split()
        loss = sum(generated_token != target_token for generated_token, target_token in zip(generated_tokens, target_tokens))
        losses.append(loss)

    loss_tensor = torch.tensor(losses, dtype=torch.float32).to(device)
    total_loss = torch.mean(loss_tensor)
    return total_loss, generated_texts

# %%
def test(dataloader, model, exact_match=True):
    total_loss = 0
    test_preds = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output1 = model(**batch, output_hidden_states=True)
            loss, preds = exact_match_loss(model, batch, output1) if exact_match else (output1.loss, test_preds)
            sequence_length_labels = batch['labels'].shape[1]
            inputs_embeds = output1.hidden_states[-1][:, -sequence_length_labels:]
            task_ids = torch.tensor([1 for i in batch["task_ids"]]).to(device)
            output2 = model(inputs_embeds=inputs_embeds, labels=batch['labels'], attention_mask=batch['attention_mask'], task_ids=task_ids, output_hidden_states=True)
            loss, preds = exact_match_loss(model, batch, output2) if exact_match else (output2.loss, test_preds)
    
        total_loss += loss.detach().float()
        test_preds.extend(preds)

    total_loss = total_loss / len(dataloader)
    return total_loss, test_preds

# %%
import os
from dln.dataset import init_dataset
from datasets import Dataset, DatasetDict

def load_dln_dataset_to_hf_dataset(data_dir, dataset_id):
    """Some gynmastics to load the dln dataset into a HuggingFace Dataset.
    dln.dataset should implement an interface compatible with HuggingFace"""

    dln_dataset = init_dataset(
        dataset_id=dataset_id,
        seed=42,
        data_dir=data_dir,
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

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# %%
@click.command()
@click.option("--data_dir", default=os.path.dirname(dln.__file__) + "/../data")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--batch_size", type=int, default=10)
@click.option("--epochs", type=int, default=50)
@click.option("--learning_rate", type=float, default=0.03)
@click.option("--num_virtual_tokens", type=int, default=16)
@click.option("--dataset", type=str, default="logical_deduction_seven_objects")
@click.option(
    "--enable_wandb",
    is_flag=False,
    help="Enable wandb logging. Requires wandb to be installed.",
)
def main(
    data_dir,
    seed,
    batch_size,
    epochs,
    learning_rate,
    num_virtual_tokens,
    dataset,
    enable_wandb,
):
    if enable_wandb:
        wandb_config = {
            "learning_rate": learning_rate,
            "architecture": "DLN",
            "dataset": dataset,
            "epochs": epochs,
            "seed": seed,
            "batch_size": batch_size,
            "num_virtual_tokens": num_virtual_tokens
        }
        wandb.init(
            project="Soft-prompt DLN",
            config=wandb_config
        )    
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

    set_seed(seed)

    model_name_or_path = "microsoft/phi-2"
    tokenizer_name_or_path = "microsoft/phi-2"

    dataset_id = dataset
    initial_instruction = (
        "Read the following question. Then, select the correct answer."
    )
    text_column = "text"
    label_column = "label"
    max_length = 128
    lr = learning_rate
    num_epochs = epochs

    peft_config = MultitaskPromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_tasks=2,
        prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=initial_instruction,
        num_transformer_submodules=1,
        tokenizer_name_or_path=model_name_or_path,
    )

    dataset = load_dln_dataset_to_hf_dataset(data_dir, dataset_id)

    classes = list(set(dataset["train"]["label"]))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, device_map="auto", padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    saved_model = None
    # try:
    #     saved_model = PeftModel.from_pretrained(model, "data/models/" + model_name_or_path + "/multitask")
    #     model = saved_model
    #     print("Using saved model from data/models/" + model_name_or_path)
    # except ValueError:
    #     model = get_peft_model(model, peft_config)
    #     print("Model not found, training new model")
    model = get_peft_model(model, peft_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler= get_linear_schedule_with_warmup(
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

    init_test_loss, test_preds = test(test_dataloader, model)
    init_test_ppl = torch.exp(init_test_loss)  # Perplexity
    print(f"Test before training: {init_test_ppl=} {init_test_loss=}")

    best_eval_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10  # Number of epochs to wait before stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output1 = model(**batch, output_hidden_states=True)
            
            sequence_length_labels = batch['labels'].shape[1]
            inputs_embeds = output1.hidden_states[-1][:, -sequence_length_labels:]
            task_ids = torch.tensor([1 for i in batch["task_ids"]]).to(device)
            output2 = model(inputs_embeds=inputs_embeds, labels=batch['labels'], attention_mask=batch['attention_mask'], task_ids=task_ids, output_hidden_states=True)
        
            loss = output1.loss + output2.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        eval_epoch_loss, eval_preds = test(eval_dataloader, model)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(train_epoch_loss))

        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )
        if enable_wandb:
            wandb.log({"training loss": train_epoch_loss, "eval accuracy": (1 - eval_epoch_loss)})

        # Sum the eval losses from all processes
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            # In the main process, prepare to receive the sum of eval losses
            total_eval_loss = torch.zeros_like(eval_epoch_loss)
        else:
            total_eval_loss = eval_epoch_loss

        if dist.is_available() and dist.is_initialized():
            dist.reduce(total_eval_loss, dst=0, op=dist.ReduceOp.SUM)
            # Broadcast the total eval loss from the main process to all other processes
            dist.broadcast(total_eval_loss, src=0)
            eval_epoch_loss = total_eval_loss / dist.get_world_size()

        # Check if the validation loss has improved
        if eval_epoch_loss < best_eval_loss:
            best_eval_loss = eval_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # if epochs_without_improvement >= patience:
        #     print("Stopping training: ", dist.get_rank())
        #     break

    model.eval()

    final_test_loss, test_preds = test(test_dataloader, model)
    final_test_ppl = torch.exp(final_test_loss)
    print(f"Test before training1: {init_test_ppl=} {init_test_loss=}")
    print(f"Test after training1: {final_test_ppl=} {final_test_loss=}")

    # if not saved_model:
    #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #         model.module.save_pretrained("data/models/" + model_name_or_path + "/multitask")
    #     else:
    #         model.save_pretrained("data/models/" + model_name_or_path + "/multitask")
    # %%
    # Ensure final_test_loss is on the correct device
    final_test_loss = final_test_loss.cuda()

    # Use dist.reduce() to add final_test_loss from all GPUs
    if dist.is_available() and dist.is_initialized():
        # The reduce operation sums all the final_test_loss and stores the result in rank 0
        dist.reduce(final_test_loss, dst=0, op=dist.ReduceOp.SUM)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Gather test_preds from all GPUs to the main GPU
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            accuracy = round(100 * (1- (final_test_loss.item() / dist.get_world_size())), 1)
            print(f"{test_preds[:10]=}")
            print(f"{dataset['test']['label'][:10]=}")
            print(f"{accuracy=}% on the test dataset")
            if enable_wandb:
                wandb.log({"test accuracy": accuracy/100})
    else:
        correct = 0
        total = 0
        for pred, label in zip(test_preds, dataset['test']['label']):
            if pred.strip() == label.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100

        print(f"{test_preds[:10]=}")
        print(f"{dataset['test']['label'][:10]=}")
        print(f"{accuracy=}% on the test dataset")
        print(f"test accuracy: {accuracy}/100")
        if enable_wandb:
            wandb.log({"test accuracy": accuracy/100})

    if enable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
