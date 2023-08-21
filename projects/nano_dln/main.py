import json
import os
import sys

from dln.loss import ZeroOneLoss
from dln.score import OutputClasses

sys.path.append("../..")  # Adds higher directory to python modules path.

import logging

import tqdm
import numpy as np
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from layers import ForwardLayer, ResidualLayer, StepLayer
from dln.operator import forward_instantiate, backward_instantiate
from dln.postprocessing import postprocess_prediction
from dln.dataset import Dataset
from utils import dumps_config, fix_seed, get_start_time, load_config, setup_logging


class PromptNet:
    def __init__(
        self,
        num_layers,
        num_prompts,
        layers_initialization,
        residual_net=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.residual_net = residual_net
        self.layers = self.initialize_layers(
            num_layers,
            layers_initialization,
        )

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x):
        output = x
        residual = None
        for layer in self.layers:
            output_res = output
            output = layer(output, residual=residual)
            residual = output_res
        return output

    def backward(self, y, loss):
        bwd = y
        weights = None
        for layer in reversed(self.layers):
            bwd, weights = layer.backward(
                bwd, weights, loss, self.num_prompts, self.num_prompts
            )
        return bwd

    def initialize_layers(
        self,
        num_layers,
        layers_initialization,
    ):
        override = {
            initialization["layer_index"]: (
                initialization.get("initial_instruction"),
                initialization.get("output_formating_instruction"),
                initialization.get("output_classes"),
            )
            for initialization in layers_initialization
        }
        layers = []

        for i in range(num_layers):
            initial_instruction, output_formating_instruction, classes = override.get(i)
            layer = ResidualLayer(
                init=initial_instruction,
                output_formating_instruction=output_formating_instruction,
                output_classes=OutputClasses(protos=classes) if classes else None,
            )
            layers.append(layer)

        layers[0].requires_input = False
        return layers


def train(args, writer):
    forward_instantiate(
        args.fwd_model,
        temperature=args.fwd_temp,
        max_tokens=args.fwd_max_tokens,
        stop=None,
    )

    backward_instantiate(
        args.bwd_model,
        temperature=args.bwd_temp,
        max_tokens=args.bwd_max_tokens,
        stop=None,
    )

    dataset = Dataset(args.data_path, args.dataset, args.seed)

    loss_fn = ZeroOneLoss(postproc=postprocess_prediction)

    model = PromptNet(
        num_layers=args.num_layers,
        num_prompts=args.num_prompts,
        layers_initialization=args.layers_initialization,
    )

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    eval_freq = args.eval_freq
    do_train = args.do_train

    best_weights = [layer.weight for layer in model.layers]
    if not args.skip_first_test_accuracy:
        best_dev_accuracy = test(
            dataset, model, loss_fn, batch_size, split="dev"
        )
        logging.info("Dec Accuracy before training: %.2f", best_dev_accuracy)
        writer.add_scalar(f"Accuracy/dev", best_dev_accuracy, 0)
    else:
        best_dev_accuracy = 0

    if do_train:  # exhaust all test data, batch by batch
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for n in tqdm.tqdm(range(eval_freq)):
                train_sentences, train_targets = dataset.get_batch(
                    "train", batch_size, random_sample=True
                )[:2]
                train_sentences = np.asarray(train_sentences)
                train_targets = np.asarray(train_targets)

                outputs = model(train_sentences)
                loss = loss_fn(outputs, train_targets)
                model.backward(train_targets, loss)

                logging.info("Current prompts:")
                for layer in model.layers:
                    logging.info("Layer: " + layer.weight)

                batch_loss = sum([l for l in loss]) / batch_size
                epoch_loss += batch_loss

                logging.info("Batch Loss: %.2f", batch_loss)
                logging.info("Running Loss: %.2f", epoch_loss / (n + 1))

            epoch_loss /= eval_freq
            dev_accuracy = test(dataset, model, loss_fn, batch_size, split="dev")

            logging.info(f"Iteration: {epoch}...")
            logging.info("Loss: %.2f", epoch_loss)
            logging.info("Dev Accuracy: %.2f", dev_accuracy)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_weights = [layer.weight for layer in model.layers]
                logging.info("Saving best weights...")

            writer.add_scalar(f"Loss/train", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/dev", dev_accuracy, epoch)

        # restore best weights
        for layer, init in zip(model.layers, best_weights):
            layer.weight = init

    test_accuracy = test(dataset, model, loss_fn, batch_size, split="test")
    logging.info(f"Test Accuracy before training: {test_accuracy_before_train}")
    logging.info(f"Test Accuracy after training: {test_accuracy}")

    writer.add_scalar(f"Accuracy/test", test_accuracy, 1)
    writer.close()


def test(dataset, model, loss_fn, batch_size=10, split="test"):
    acc = 0.0
    gt_labels, pred_labels = [], []

    pbar = tqdm.tqdm(total=dataset.get_size(split))
    for i, test_batch in enumerate(
        dataset.iterate(split, batch_size, random_sample=False)
    ):
        test_sentences, test_targets = test_batch[:2]
        batch_size = len(test_targets)
        outputs = model(test_sentences)
        loss = loss_fn(outputs, test_targets)
        acc = 1.0 - sum([l for l in loss]) / batch_size

        gt_labels += [postprocess_prediction(item) for item in test_targets]
        pred_labels += [postprocess_prediction(item) for item in outputs]

        pbar.update(batch_size)
        pbar.set_description_str(
            "Running Acc: {:.3f}".format(
                (np.array(gt_labels) == np.array(pred_labels)).mean()
            )
        )

    accuracy = np.mean(
        [1 if _p.lower() == _g else 0 for _p, _g in zip(pred_labels, gt_labels)]
    )
    return accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--score_method", type=str, default="rank_over_examples")
    parser.add_argument("--dataset", type=str, default="subj")
    parser.add_argument("--fwd_model", type=str, default="gpt3/text-davinci-003")
    parser.add_argument("--bwd_model", type=str, default="gpt3/text-davinci-003")
    parser.add_argument("--fwd_max_tokens", type=int, default=128)
    parser.add_argument("--bwd_max_tokens", type=int, default=128)
    parser.add_argument("--fwd_temp", type=float, default=0.0)
    parser.add_argument("--bwd_temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./log", help="log directory")
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Log verbosity. Options: debug, info, warning, error, critical.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="BackwardsOperator",
        help="Name of the run. Used for logging and tensorboard.",
    )
    parser.add_argument(
        "--skip_first_test_accuracy", action="store_true", default=False
    )
    parser.add_argument(
        "--default_initial_instruction",
        type=str,
        default=None,
        help="Initial instruction to use if not specified in layers_initialization.",
    )
    parser.add_argument(
        "--default_output_formating_instruction",
        type=str,
        default=None,
        help="Output formating instruction to use if not specified in layers_initialization.",
    )
    parser.add_argument("--layers_initialization", type=json.loads, default={})
    parser.add_argument(
        "--residual_net", action="store_true", default=False
    )  # this could also goes into layers_initialization

    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        parser.set_defaults(**config)
        args = parser.parse_args()

    config_vars = vars(args)
    config_txt = [args.experiment_name]
    config_txt += [
        f"{key}_{config_vars[key]}"
        for key in ("num_layers", "num_prompts", "score_method", "batch_size")
    ]
    config_dir = "_".join(config_txt)
    start_time_dir = get_start_time()
    setup_logging(
        args.log_level, f"{args.log_dir}/{args.dataset}/{config_dir}/{start_time_dir}"
    )
    writer = SummaryWriter(
        log_dir=os.path.join(
            args.log_dir, args.dataset, "tensorboard", config_dir, start_time_dir
        )
    )
    dumps_config(
        config_vars, os.path.join(args.log_dir, args.dataset, "tensorboard", config_dir)
    )

    logging.info(args)
    fix_seed(args.seed)
    train(args, writer)
