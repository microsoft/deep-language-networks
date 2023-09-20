import json
import os
import sys

sys.path.append("../..")  # Adds higher directory to python modules path.

import logging
import tqdm
import numpy as np
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from engine.layers import ResidualLayer
from engine.scorer import OutputClasses
from engine.ops import LanguageLayerOps
from engine.configs import BackpropLogProbsEngine, BackwardLogProbsEngine
from engine.loss import ZeroOneLoss
from postprocessing import postprocess_prediction
from dataset import Dataset
from utils import dumps_config, fix_seed, get_start_time, load_config, setup_logging


class PromptNet:
    def __init__(
        self,
        num_prompts,
        num_hiddens,
        layers_initialization,
        logp_penalty=1.0,
        memory_size=5,
        engine="backward_logprobs",
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.num_hiddens = num_hiddens
        self.logp_penalty = logp_penalty
        self.memory_size = memory_size
        self.engine = engine
        self.layers = self.initialize_layers(
            layers_initialization,
        )

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x):
        return self.layers[0].forward_graph(x)

    def backward(self, y, losses):
        bwd = y
        weights = None
        for layer in reversed(self.layers):
            bwd, weights = layer.backward(
                bwd,
                weights,
                losses,
                targets=y,
                num_p_samples=self.num_prompts,
                num_h_samples=self.num_hiddens
            )
        return bwd

    def initialize_layers(
        self,
        layers_initialization,
    ):
        override = {
            initialization["layer_index"]: (
                initialization.get("template_type"),
                initialization.get("initial_instruction"),
                initialization.get("output_formatting_instruction"),
                initialization.get("output_classes"),
            )
            for initialization in layers_initialization
        }

        layers = []
        for i in range(len(layers_initialization)):
            (
                template_type,
                initial_instruction,
                output_formatting_instruction,
                classes,
            ) = override.get(i)

            if self.engine == "backward_logprobs":
                engine = BackwardLogProbsEngine(
                    memory_size=self.memory_size,
                    logp_penalty=self.logp_penalty
                )
            elif self.engine == "backprop_logprobs":
                engine = BackpropLogProbsEngine(
                    memory_size=self.memory_size,
                    logp_penalty=self.logp_penalty
                )
            else:
                raise ValueError(f"Unknown engine: {self.engine}")

            layer = (
                ResidualLayer(
                    template_type=template_type,
                    init=initial_instruction,
                    output_formatting_instruction=output_formatting_instruction,
                    output_classes=OutputClasses(protos=classes) if classes else None,
                ).with_engine(engine)
            )
            layers.append(layer)
            if i > 0:
                layers[i - 1].connect_to(layer)
        layers[0].requires_input = False
        return layers


def train(args, writer):
    LanguageLayerOps().instantiate_forward_lm(
        args.fwd_model,
        temperature=args.fwd_temp,
        max_tokens=args.fwd_max_tokens,
        stop=None,
    )
    LanguageLayerOps().instantiate_backward_lm(
        args.bwd_model,
        temperature=args.bwd_temp,
        max_tokens=args.bwd_max_tokens,
        stop=None,
    )
    if args.sco_model:
        LanguageLayerOps().instantiate_scoring_lm(
            args.sco_model,
            temperature=args.bwd_temp,
            max_tokens=args.bwd_max_tokens,
            stop=None,
        )

    dataset = Dataset(
        args.data_path,
        args.dataset,
        args.seed,
        num_train_examples=args.num_train_examples,
        num_dev_examples=args.num_dev_examples,
    )

    loss_fn = ZeroOneLoss(postproc=postprocess_prediction)
    model = PromptNet(
        num_prompts=args.num_prompts,
        num_hiddens=args.num_hiddens,
        layers_initialization=args.layers_initialization,
        logp_penalty=args.logp_penalty,
        memory_size=args.memory_size,
        engine=args.engine,
    )

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    eval_freq = args.eval_freq
    do_train = args.do_train

    best_weights = [layer.weight for layer in model.layers]
    if not args.skip_first_test_accuracy:
        dev_accuracy_before_training = best_dev_accuracy = test(
            dataset, model, loss_fn, batch_size, split="dev"
        )
        test_accuracy_before_training = test(
            dataset, model, loss_fn, batch_size, split="test"
        )
        logging.info("Dec Accuracy before training: %.2f", best_dev_accuracy)
        logging.info(
            "Test Accuracy before training: %.2f", test_accuracy_before_training
        )
        writer.add_scalar(f"Accuracy/dev", best_dev_accuracy, 0)
        writer.add_scalar(f"Accuracy/test", test_accuracy_before_training, 0)
    else:
        best_dev_accuracy = 0
        test_accuracy_before_training = 0

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
                losses = loss_fn(outputs, train_targets)

                train_targets = np.array([loss_fn.postproc(t) for t in train_targets])
                model.backward(train_targets, losses)

                logging.info("Current prompts:")
                for layer in model.layers:
                    logging.info("Layer: " + layer.weight)

                batch_loss = sum([l for l in losses]) / batch_size
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
                logging.info("Best Dev Accuracy: %.2f", best_dev_accuracy)
                logging.info("Saving best weights...")

            writer.add_scalar(f"Loss/train", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/dev", dev_accuracy, epoch)

        # restore best weights
        for layer, init in zip(model.layers, best_weights):
            layer.weight = init

    test_accuracy = test(dataset, model, loss_fn, batch_size, split="test")

    logging.info(f"Test Accuracy before training: {test_accuracy_before_training}")
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
    parser.add_argument("--logp_penalty", type=float, default=1.)
    parser.add_argument("--memory_size", type=int, default=5)
    parser.add_argument("--engine", type=str, default="backward_logprobs")
    parser.add_argument("--num_prompts", type=int, default=5)
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
        "--default_output_formatting_instruction",
        type=str,
        default=None,
        help="Output formating instruction to use if not specified in layers_initialization.",
    )
    parser.add_argument("--layers_initialization", type=json.loads, default={})

    args = parser.parse_args()
    if args.config is not None:
        config = load_config(args.config)
        parser.set_defaults(**config)
        args = parser.parse_args()

    config_vars = vars(args)
    config_txt = [args.experiment_name]
    config_txt += [
        f"{key}={config_vars[key]}"
        for key in (
            "num_prompts",
            "num_hiddens",
            "batch_size",
            "seed",
            "logp_penalty",
            "engine",
            "memory_size",
        )
    ]
    config_dir = "-".join(config_txt)

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
