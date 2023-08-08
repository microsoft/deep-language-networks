import datetime
import json
import logging
import os
from collections import Counter

import click
import numpy as np
import tqdm
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from dln.dataset import init_dataset
from dln.loss import ZeroOneLoss
from dln.operator import backward_instantiate, forward_instantiate
from dln.postprocessing import postprocess_prediction
from dln.vi.model import VILModel, log_message


def init_prompts(dataset, init_p1, init_p2):
    """Initialize the prompts for the two layers of the model.
    If init_p1 or init_p2 is a json file, load the best weights from the json file.
    """

    if init_p1 and init_p1.endswith(".json"):
        with open(init_p1) as f:
            best_weights = json.load(f)
        init_p1 = best_weights[dataset.name]["best_weights"]
    elif init_p2 and init_p2.endswith(".json"):
        with open(init_p2) as f:
            best_weights = json.load(f)
        init_p2 = best_weights[dataset.name]["best_weights"]
    elif init_p2 and init_p2.endswith(".log"):
        found = False
        with open(init_p2) as f:
            lines = f.readlines()
            for line in lines:
                if "Best L2 weights" in line:
                    init_p2 = line.partition("Best L2 weights:")[-1].strip()
                    found = True
                    break
            if not found:
                raise ValueError("Best weights were not found in the log file!")
    if init_p2 is None:
        init_p2 = dataset.instruction
    return init_p1, init_p2


def validate(dataset, model, loss_fn, iteration, val_examples, val_scores, writer):
    log_message(colored("VALIDATING...", "red"))
    log_message("Current L1 weights:", model.encoder_l1.weight)
    log_message("Current L2 weights:", model.encoder_l2.weight)

    val_key = "{}-{}".format(model.encoder_l1.weight, model.encoder_l2.weight)
    if val_key in val_scores:
        log_message("Already evaluated this configuration, skipping...")
        dev_acc = val_scores[val_key]
    else:
        acc = 0.0
        tot = 0.0
        pbar = tqdm.tqdm(
            total=dataset.get_size("dev") if val_examples < 0 else val_examples,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            desc="Eval",
        )
        dataset.reset_pointer("dev")
        num_examples = 0
        class_counter = Counter()
        total_counter = Counter()

        for batch in dataset.iterate("dev", batch_size=20):
            x, y, infos = batch
            y_hat = model.forward(np.array(x), infos=infos)
            losses = loss_fn(y_hat, y)
            acc += len(y) - np.sum(losses)
            tot += len(y)
            num_examples += len(y)

            for xi, yi, yhati, li in zip(x, y, y_hat, losses):
                total_counter.update([yi])
                if li > 0:
                    class_counter.update([yi])

            pbar.update(len(y))
            pbar.set_postfix_str(f"{acc / tot:.1%}")

            if num_examples == val_examples:
                break

        for k, v in class_counter.items():
            log_message(f"{k}: {float(v)/total_counter[k]}")
        dev_acc = acc / tot
        val_scores[val_key] = dev_acc

    if iteration == 0:
        log_message(colored("INIT DEV ACC: {}".format(dev_acc), "red"))
    log_message(colored("DEV ACC: {}".format(dev_acc), "red"))
    writer.add_scalar("dev/acc", (dev_acc), iteration)
    return dev_acc


def test(dataset, model, loss_fn, iteration, writer, cost_only=False):
    log_message(colored("TESTING...", "red"))
    acc = 0.0
    tot = 0.0
    all_accs = []

    pbar = tqdm.tqdm(
        total=dataset.get_size("test"),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Eval",
    )

    model.cost = 0.
    dataset.reset_pointer("test")
    for batch in dataset.iterate("test", batch_size=20):
        x, y, infos = batch
        y_hat = model.forward(np.array(x), infos=infos, cost_only=cost_only)
        all_accs += (1. - loss_fn(y_hat, y)).tolist()
        acc += len(y) - np.sum(loss_fn(y_hat, y))
        tot += len(y)
        pbar.update(len(y))
        pbar.set_postfix_str(f"{acc / tot:.1%}")

    test_acc = acc / tot
    writer.add_scalar("test/acc", (test_acc), iteration)
    # for sig-test purposes
    log_message("ALL ACCS:", all_accs)
    log_message("TOKEN COST:", model.cost)
    return test_acc


@click.command()
@click.option("--seed", default=42, help="Random seed.")
@click.option("--out_dir", default="log/")
@click.option("--data_dir", default="../../data")
@click.option("--val_freq", default=2)
@click.option("--do_first_eval", is_flag=True)
@click.option("--do_zero_shot", is_flag=True)
@click.option("--do_few_shot", default=-1, type=int)
@click.option("--q_hidden", default="suffix_forward_tbs")
@click.option("--q_prompt", default="q_action_prompt")
@click.option("--p_hidden", default="suffix_forward_tbs")
@click.option("--p_class", default="classify_forward")
@click.option("--p_residual", type=str, default="classify_residual")
@click.option("--balance_batch", is_flag=True, help="Balance batch.")
@click.option("--batch_size", type=int, default=20)
@click.option("--one_layer", is_flag=True)
@click.option("--dataset", type=str, default="subj")
@click.option("--use_h_argmax", type=bool, default=False)
@click.option("--iters", type=int, default=20)
@click.option("--num_p_samples", type=int, default=5)
@click.option("--num_h_samples", type=int, default=3)
@click.option("--tolerance", type=int, default=-1)
@click.option("--compute_cost", is_flag=True)
@click.option(
    "--strip_options_for_hidden",
    type=bool,
    default=False,
    help="Remove options from examples for the hidden layer.",
)
@click.option(
    "--strip_prefix_for_hidden",
    type=bool,
    default=False,
    help="Strip the prefix from the examples if it exists in some tasks, e.g. BBH.",
)
@click.option(
    "--strip_answer_for_hidden",
    type=bool,
    default=False,
    help="Strip the 'Answer:' from the hidden state, if the model generates it.",
)
@click.option(
    "--trust_factor",
    default=0.0,
    help="Trust-region factor for prompt update. Ensures KL divergence between the old and new prompt is small.",
)
@click.option(
    "--fwd_temp",
    default=0.0,
    help="Forward temperature",
)
@click.option(
    "--bwd_temp",
    default=0.7,
    help="Backward temperature",
)
@click.option(
    "--use_memory",
    type=int,
    default=0,
    help="Include evaluation of past prompts that have worked well in the selection list.",
)
@click.option(
    "--forward_use_classes",
    type=bool,
    default=False,
    help="Uses classes in the forward pass, constrains the output space.",
)
@click.option(
    "--init_p1",
    type=str,
    default="Decompose the problem to make it simpler:",
)
@click.option(
    "--init_p2",
    type=str,
    default=None,
)
@click.option(
    "--held_out_prompt_ranking",
    type=bool,
    default=False,
    help="Evaluate prompts to keep for the next iteration on held-out examples in the current batch.",
)
@click.option(
    "--train_p1",
    type=bool,
    default=True,
    help="Train 1 layer, if False, keep it fixed.",
)
@click.option(
    "--train_p2",
    type=bool,
    default=True,
    help="Train 2 layer, if False, keep it fixed.",
)
@click.option(
    "--logp_penalty",
    type=float,
    default=0.0,
    help="Logp penalty for hiddens that haven't worked. Encourages exploration.",
)
@click.option(
    "--decay_logp_penalty",
    type=bool,
    default=True,
    help="Decay logp penalty linearly, reaching zero at the last iteration.",
)
@click.option(
    "--output_scoring_function",
    type=str,
    default="logprobs",
    help="Use logprobs to score output predictions.",
)
@click.option(
    "--hidden_scoring_function",
    type=str,
    default="logprobs",
    help="Use logprobs to score hidden states",
)
@click.option(
    "--posterior_temp",
    type=float,
    default=1.0,
    help="Sharpen (<1.0)/Flatten (>1.0) the posterior distribution over h.",
)
@click.option(
    "--model_type",
    type=str,
    default="text-davinci-003",
)
@click.option(
    "--bwd_model_type",
    type=str,
    default=None,
)
@click.option(
    "--fwd_max_tokens",
    type=int,
    default=256,
    help="Forward max tokens.",
)
@click.option(
    "--bwd_max_tokens",
    type=int,
    default=512,
    help="Backward max tokens.",
)
@click.option(
    "--num_p1_steps",
    type=int,
    default=1,
    help="Number of prompt optimization steps for the hidden layer.",
)
def main(
    seed,
    out_dir,
    data_dir,
    val_freq,
    compute_cost,
    do_first_eval,
    do_zero_shot,
    do_few_shot,
    q_hidden,
    q_prompt,
    p_hidden,
    p_class,
    p_residual,
    fwd_temp,
    bwd_temp,
    balance_batch,
    batch_size,
    one_layer,
    dataset,
    use_h_argmax,
    iters,
    num_p_samples,
    num_h_samples,
    strip_options_for_hidden,
    strip_answer_for_hidden,
    strip_prefix_for_hidden,
    trust_factor,
    use_memory,
    init_p1,
    init_p2,
    tolerance,
    output_scoring_function,
    hidden_scoring_function,
    forward_use_classes,
    held_out_prompt_ranking,
    train_p1,
    train_p2,
    logp_penalty,
    decay_logp_penalty,
    posterior_temp,
    model_type,
    bwd_model_type,
    fwd_max_tokens,
    bwd_max_tokens,
    num_p1_steps,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    out_dir = f"{out_dir}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{out_dir}/output.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_message(json.dumps(locals()))
    log_message("Logging to... {}".format(out_dir + "/output.log"))

    writer = SummaryWriter(f"{out_dir}")

    dataset, output_classes, val_examples = init_dataset(dataset, seed, data_dir, do_few_shot)

    init_p1, init_p2 = init_prompts(dataset, init_p1, init_p2)
    log_message("Init P1: ", init_p1)
    log_message("Init P2: ", init_p2)

    forward_instantiate(
        model_type,
        temperature=0.0,
        max_tokens=fwd_max_tokens,
        stop=None,
    )
    backward_instantiate(
        bwd_model_type or model_type,
        temperature=bwd_temp,
        max_tokens=bwd_max_tokens,
        stop=None,
    )

    loss_fn = ZeroOneLoss(postproc=postprocess_prediction)
    model = VILModel(
        loss_fn,
        init_p1=init_p1,
        init_p2=init_p2,
        two_layers=not one_layer,
        num_p_samples=num_p_samples,
        num_h_samples=num_h_samples,
        q_hidden=q_hidden,
        q_prompt=q_prompt,
        p_hidden=p_hidden,
        p_class=p_class,
        p_residual=p_residual,
        use_h_argmax=use_h_argmax,
        output_classes=output_classes,
        strip_options_for_hidden=strip_options_for_hidden,
        strip_answer_for_hidden=strip_answer_for_hidden,
        trust_factor=trust_factor,
        forward_use_classes=forward_use_classes,
        held_out_prompt_ranking=held_out_prompt_ranking,
        use_memory=use_memory,
        train_p1=train_p1,
        train_p2=train_p2,
        logp_penalty=logp_penalty,
        p1_max_tokens=256,
        p2_max_tokens=20,
        posterior_temp=posterior_temp,
        strip_prefix_for_hidden=dataset.prefix if strip_prefix_for_hidden else None,
        output_scoring_function=output_scoring_function,
        hidden_scoring_function=hidden_scoring_function,
        num_p1_steps=num_p1_steps,
    )

    running_acc = 0.0
    running_elbo = 0.0
    best_dev = 0.0
    best_ps = [model.encoder_l1.weight, model.encoder_l2.weight]
    val_scores = {}

    patience = 0
    for iteration in range(iters + 1):
        log_message("STARTING EPOCH {} - {}".format(iteration, out_dir))

        if iteration % val_freq == 0 and (
            iteration > 0 or do_first_eval
        ):
            dev_acc = validate(
                dataset, model, loss_fn, iteration, val_examples, val_scores, writer
            )
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_ps = (model.encoder_l1.weight, model.encoder_l2.weight)

                if use_memory:
                    model.add_to_memory(*best_ps, score=best_dev)

                log_message(colored("BEST DEV ACC: {}".format(best_dev), "red"))
                patience = 0
            else:
                patience += 1

            if tolerance >= 0 and patience >= tolerance:
                log_message("Loading back the best model...")
                model.encoder_l1.weight = best_ps[0]
                model.encoder_l2.weight = best_ps[1]
                patience = 0

        # zero shot or allow last iteration for validation
        if do_zero_shot or iteration == iters or (do_few_shot >= 0 and not train_p1 and not train_p2):
            break

        x, y, infos = dataset.get_batch(
            "train", batch_size, random_sample=True, balance=balance_batch
        )

        if decay_logp_penalty:
            model.logp_penalty = logp_penalty * (1.0 - (iteration / iters))

        log_message(colored("Training P2? {}".format(model.train_p2), "red"))
        log_message(colored("LOGPenalty? {}".format(model.logp_penalty), "red"))
        elbo, p1, p2, loss, elbo1, elbo2 = model.forward(
            np.array(x), np.array(y), infos=infos, temperature=fwd_temp
        )

        # Update prompts
        model.encoder_l1.weight = p1
        model.encoder_l2.weight = p2
        log_message("Patience: {}".format(patience))

        if iteration == 0:
            running_elbo = elbo
            running_acc = 1.0 - loss
        else:
            running_elbo = 0.2 * elbo + 0.8 * running_elbo
            running_acc = 0.2 * (1.0 - loss) + 0.8 * running_acc

        log_message("--------------------")
        log_message(colored("{} TRAINING EPOCH DONE.".format(iteration), "blue"))
        log_message(colored("ELBO: {}".format(elbo), "blue"))
        log_message(colored("ACC: {}".format((1.0 - loss)), "blue"))
        log_message(colored("RUN ELBO: {}".format(running_elbo), "blue"))
        log_message(colored("RUN ACC: {}".format(running_acc), "blue"))
        log_message(colored("BATCH Y BALANCE: {}".format(Counter(y)), "blue"))
        log_message(colored("BATCH X LEN: {}".format([len(x_i) for x_i in x]), "blue"))

        writer.add_scalar("elbo", elbo, iteration)
        writer.add_scalar("elbo1", elbo1, iteration)
        writer.add_scalar("elbo2", elbo2, iteration)
        writer.add_scalar("acc", (1.0 - loss), iteration)

    log_message("--------------------")
    log_message("Loading best model...")

    model.encoder_l1.weight = best_ps[0]
    model.encoder_l2.weight = best_ps[1]

    log_message("Best L1 weights:", model.encoder_l1.weight)
    log_message("Best L2 weights:", model.encoder_l2.weight)

    test_acc = test(dataset, model, loss_fn, iteration, writer, cost_only=compute_cost)

    log_message(colored("DEV ACC: {}".format(best_dev), "green"))
    log_message(colored("TEST ACC: {}".format(test_acc), "green"))
    writer.close()


if __name__ == "__main__":
    main()
