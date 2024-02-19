import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
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


def exact_match_loss(generated_texts, target_texts):
    losses = []
    for generated_text, target_text in zip(generated_texts, target_texts):
        generated_tokens = generated_text.split()
        target_tokens = target_text.split()
        loss = sum(generated_token != target_token for generated_token, target_token in zip(generated_tokens, target_tokens))
        losses.append(loss)
    
    loss_tensor = torch.tensor(losses, dtype=torch.float32)
    total_loss = torch.mean(loss_tensor)
    
    print(generated_texts)
    print(target_texts)
    print(total_loss.item())
    
    return total_loss

def test(dataloader, model, tokenizer, device, exact_match=False):
    loss = 0
    preds = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"], max_length=500, num_return_sequences=1) if exact_match else model(**batch)
        
        if exact_match:
            generated_texts = tokenizer.batch_decode(outputs,  skip_special_tokens=True) #[tokenizer.decode(out, skip_special_tokens=True) for out in outputs]        
            target_texts_decoded = [tokenizer.decode(target, skip_special_tokens=True) for target in batch["labels"]]

        loss = exact_match_loss(generated_texts, target_texts_decoded) if exact_match else outputs.loss
        loss += loss.detach().float()
        # preds.extend(
        #     tokenizer.batch_decode(
        #         torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
        #         skip_special_tokens=True,
        #     )
        # )
        labels = torch.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)

        # targets = []
        # for label_row in labels:
        #     decoded_tokens = tokenizer.convert_ids_to_tokens(label_row, skip_special_tokens=True)
        #     decoded_text = tokenizer.convert_tokens_to_string(decoded_tokens)
        #     targets.append(decoded_text)

        # if (exact_match):
        #     print(preds)
            # print(targets)

    loss = loss / len(dataloader)
    return loss


def preprocess_function(examples, tokenizer, prefix, text_column, label_column, max_length):
    batch_size = len(examples[text_column])
    inputs = [f"{prefix}\n\n{x}\n\nAnswer:\n" for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        # concat the inputs and labels, mask the inputs part, and update the
        # attention mask to match the new length (inputs + labels + pad_token_id)
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # masks / ignores -100 tokens in the loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
        labels["input_ids"][i] = [tokenizer.pad_token_id] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        # pad or truncate the batch to the specified max_length, and update the attention mask
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (
            max_length - len(sample_input_ids)
        ) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(
            model_inputs["input_ids"][i][:max_length]
        )
        model_inputs["attention_mask"][i] = torch.tensor(
            model_inputs["attention_mask"][i][:max_length]
        )
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    accelerator = Accelerator()
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
    num_epochs = 10
    # batch_size = 8
    batch_size = 16

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text=initial_instruction,
        tokenizer_name_or_path=model_name_or_path,
    )

    dataset = load_dln_dataset_to_hf_dataset(dataset_id)

    classes = list(set(dataset["train"]["label"]))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, device_map="auto")
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
            "prefix": initial_instruction,
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
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

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

            output = model.generate(batch["input_ids"], max_length=500, num_return_sequences=1)

            generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in output]    
            target_texts_decoded = [tokenizer.decode(target, skip_special_tokens=True) for target in batch["labels"]]

            loss = exact_match_loss(generated_texts, target_texts_decoded)
            # optimizer.zero_grad()
            loss.requires_grad_(True)
            
            total_loss += loss.detach().float()
            optimizer.zero_grad()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            # optimizer.zero_grad()

        model.eval()
        eval_epoch_loss = test(eval_dataloader, model, tokenizer, device)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )

    model.eval()
    final_test_loss = test(test_dataloader, model, tokenizer, device)
    final_test_ppl = torch.exp(final_test_loss)
    print(f"Test before training: {init_test_ppl=} {init_test_loss=}")
    print(f"Test after training: {final_test_ppl=} {final_test_loss=}")

    model.module.save_pretrained("data/models/" + model_name_or_path)

    config = PeftConfig.from_pretrained("data/models/" + model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    lora_model = PeftModel.from_pretrained(model, "data/models/" + model_name_or_path)
    lora_model.to(device)

    # final_test_loss = test(test_dataloader, lora_model, tokenizer, device, True)
    # final_test_ppl = torch.exp(final_test_loss)

    # print(f"Test after loading: {final_test_ppl=} {final_test_loss=}")

    sentences = ["Read the following sentence, then determine whether you return to the starting point.\n\nIf you follow these instructions, do you return to the starting point? Take 9 steps. Take 9 steps. Take 4 steps. Turn right.\nOptions:\n- Yes\n- No\n\nAnswer:\n"]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    generate_ids = lora_model.generate(**inputs, max_length=500)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print([print(o, "\n") for o in outputs])

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
