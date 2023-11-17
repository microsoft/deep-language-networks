from argparse import ArgumentParser
import datetime
import logging
import os

from dln.dataset import Dataset, init_dataset

from dln.loss import LossRegistry
from dln.operator import LLMRegistry

from dln.vi.model import log_message
from layers import DeepWide


# try:
#     from wandb.integration.openai import autolog
#     autolog({"project":"dwln"})
# except ImportError:
#     pass


def train(model, dataset: Dataset, loss_fn, batch_size, iters):
    for _ in range(iters):
        x, y, _ = dataset.get_batch("train", batch_size, random_sample=True)
        y_hat = model.forward(x)
        loss = loss_fn(y_hat, y)
        [
            print(f"y_hat: {a}\n    y: {b}\n loss: {c}\n")
            for a, b, c in zip(y_hat, y, loss)
        ]
        model.backward(loss)


def train_dwln(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    output_log_dir = os.path.join(out_dir, "output.log")
    logging.basicConfig(
        filename=output_log_dir,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_message(str(args))
    log_message(f"Logging to... {output_log_dir}")

    dataset = init_dataset(
        dataset_id=args.dataset,
        seed=args.seed,
        data_dir=args.data_dir,
        max_train_size=args.max_train_size,
        max_dev_size=args.max_dev_size,
        max_test_size=args.max_test_size,
    )

    llm_registry = LLMRegistry.from_yaml(args.config)
    fwd_model = llm_registry[args.fwd_model]
    bwd_model = llm_registry[args.bwd_model]

    loss_fn = LossRegistry.instantiate(args.loss_function)
    dw = DeepWide(fwd_model, bwd_model)

    train(
        model=dw,
        dataset=dataset,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        iters=args.iters
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--fwd_model", type=str)
    parser.add_argument("--bwd_model", type=str)
    parser.add_argument("--loss_function", type=str)
    parser.add_argument("--output_scoring_function", type=str)
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max_train_size", type=int, default=200)
    parser.add_argument("--max_dev_size", type=int, default=200)
    parser.add_argument("--max_test_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./log", help="log directory")
    args = parser.parse_args()
    train_dwln(args)


if __name__ == "__main__":
    main()

# python main.py --config llm_config.yaml --fwd_model gpt-3-fwd --bwd_model gpt-3-bwd --dataset gsm8k --loss_function number_presence_loss --output_scoring_function accuracy --out_dir log/debug
