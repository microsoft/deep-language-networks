from collections import Counter

import numpy as np
from termcolor import colored

from dln.postprocessing import postprocess_prediction
from dln.loss import LLoss
from dln.operator import LLM
from dln.score import LogProbsScore, OutputClasses
from dln.vi.layers import PriorLayer, ResidualPriorLayer
from dln.vi.sampler import PosteriorSampler, PromptSampler
from dln.vi.utils import compute_pairwise_kl, log_message, ResultLogEntry


class VILModel:
    def __init__(
        self,
        loss_fn: LLoss,
        init_p1: str = "",
        init_p2: str = "",
        two_layers=True,
        num_h_samples: int = 3,
        num_p_samples: int = 5,
        use_h_argmax: bool = False,
        forward_evaluate: LLM = None,
        prompt_sampler_1: PromptSampler = None,
        prompt_sampler_2: PromptSampler = None,
        posterior_sampler : PosteriorSampler = None,
        logprobs_score: LogProbsScore = None,
        p_hidden: str = "suffix_forward_tbs:latest",
        p_class: str = "classify_forward:latest",
        p_residual: str = "classify_residual:latest",
        output_classes: OutputClasses = None,
        strip_options_for_hidden: bool = False,
        trust_factor: float = 0.0,
        forward_use_classes: bool = False,
        held_out_prompt_ranking: bool = False,
        use_memory: int = 0,
        train_p1: bool = True,
        train_p2: bool = True,
        logp_penalty: float = 0.0,
        p1_max_tokens: int = 256,
        p2_max_tokens: int = 20,
        posterior_temp: float = 1.0,
        strip_prefix_for_hidden: str = None,
        output_scoring_function: str = "logprobs",
        hidden_scoring_function: str = "logprobs",
        posterior_sharpening_include_prior: bool = True,
        posterior_sharpening_use_mi_regularization: bool = False,
        num_p1_steps: int = 1,
        use_nce: bool = False,
    ):
        """
        Args:
            loss_fn: loss function to use
            init_p1: initialization for the first prompt
            init_p2: initialization for the second prompt
            two_layers: whether to use two layers or one layer
            num_h_samples: number of posterior samples to use for the hidden state
            num_p_samples: number of posterior samples to use for the prompt
            use_h_argmax: whether to use the argmax of the posterior distribution when selecting best prompts, if False, then
                          we compute num_h_samples * num_p_samples scores and select prompts based on the sum of the num_h_samples scores
            forward_evaluate: LLM for the forward pass
            prompt_sampler_1: posterior sampler over the prompt
            prompt_sampler_2: posterior sampler over the hidden state
            posterior_sampler: sample hidden states from the posterior distribution
            logprobs_score: logprobs scoring function
            p_hidden: forward template for the forward pass that generates the hidden state
            p_class: forward template for the classification layer
            p_residual: forward template for the residual layer
            output_classes: if specified, we compute log-likelihood over these classes only
            strip_options_for_hidden: whether to strip the options from the input when computing the hidden state, don't use it.
            trust_factor: trust factor for the KL divergence between the current prompt and the new prompt, it acts *only* at the last layer, a sort of step size.
            forward_use_classes: whether to use the classes in the forward pass, if True, then we pick the class with the highest probability.
            held_out_prompt_ranking: when proposing the prompts from the posterior, we only use HALF of the batch, kind of limiting over-fitting, but it decreases batch size
                                     for posterior distribution.
            use_memory: whether to use memory, if 0, we don't use memory, if n, we include n best DEV prompts in the list of candidate prompts to select from, etc...
            train_p1: whether to train the first prompt
            train_p2: whether to train the second prompt
            logp_penalty: penalizes the log-likelihood of wrong thoughts
            p1_max_tokens: max tokens for the residual layer
            p2_max_tokens: max tokens for the prior layer
            posterior_temp: posterior temperature
            strip_prefix_for_hidden: strip prefix from the hidden state if the model generates it
            output_scoring_function: output scoring function, either "logprobs" or "accuracy"
            hidden_scoring_function: hidden scoring function, only "logprobs" is supported
            posterior_sharpening_include_prior: include prior in the posterior sharpening
            posterior_sharpening_use_mi_regularization: use MI regularization in the posterior sharpening
            num_p1_steps: number of optimization steps for p1
            use_nce: compute p1 elbo using NCE
        """
        self.encoder_l1 = ResidualPriorLayer(
            logprobs_score=logprobs_score,
            forward_evaluate=forward_evaluate,
            forward_template=p_hidden,
            init=init_p1,
            residual_template=p_residual,
        )
        self.encoder_l2 = PriorLayer(
            logprobs_score=logprobs_score,
            forward_evaluate=forward_evaluate,
            forward_template=p_class,
            init=init_p2,
        )
        if not two_layers:
            self.encoder_l1.weight = ""

        self.prompt_sampler_1 = prompt_sampler_1
        self.prompt_sampler_2 = prompt_sampler_2
        self.q_sampler = posterior_sampler
        self.trust_factor = trust_factor
        self.strip_options_for_hidden = strip_options_for_hidden
        self.strip_prefix_for_hidden = strip_prefix_for_hidden
        self.output_classes = output_classes
        self.two_layers = two_layers
        self.loss_fn = loss_fn
        self.num_h_samples = num_h_samples
        self.num_p_samples = num_p_samples
        self.use_h_argmax = use_h_argmax
        self.forward_use_classes = forward_use_classes
        self.held_out_prompt_ranking = held_out_prompt_ranking
        self.use_memory = use_memory
        self.train_p1 = train_p1
        self.train_p2 = train_p2
        self.logp_penalty = logp_penalty
        self.p1_max_tokens = p1_max_tokens
        self.p2_max_tokens = p2_max_tokens
        self.posterior_temp = posterior_temp
        self.num_p2_steps = 1
        self.num_p1_steps = num_p1_steps
        self.output_scoring_function = output_scoring_function
        self.hidden_scoring_function = hidden_scoring_function
        self.posterior_sharpening_include_prior = posterior_sharpening_include_prior
        self.posterior_sharpening_use_mi_regularization = posterior_sharpening_use_mi_regularization
        self.num_acc_mc_samples = 1
        self.cost = 0.0
        self.use_nce = use_nce

        if self.forward_use_classes:
            assert (
                self.output_classes is not None
            ), "Cannot use classes for forward without output classes"

        self.prompt_memory = []
        self.result_entry = ResultLogEntry()

    def get_from_memory(self, layer_index=0):
        assert layer_index in [0, 1], "Layer index out of bounds"
        return np.asarray([p[layer_index] for p in self.prompt_memory])

    def add_to_memory(self, p1, p2, score):
        """
        Max memory size is 2. Add (p1, p2, score) to memory and keep memory sorted.
        Keep best two prompts in memory.
        """
        if self.use_memory == 0:
            raise ValueError("Cannot add to memory if use_memory is 0")

        self.prompt_memory.append((p1, p2, score))
        self.prompt_memory = sorted(
            self.prompt_memory, key=lambda x: x[2], reverse=True
        )
        if len(self.prompt_memory) > self.use_memory:
            self.prompt_memory = self.prompt_memory[: self.use_memory][::-1]

    def inference_one_layer(
        self,
        x: np.array,
        y: np.array,
        y_hat: np.array,
        losses: np.array,
    ):
        batch_size = y.shape[0]

        p_tilde_2: np.array = self.prompt_sampler_2.sample_q_p(
            x,
            y,
            y_hat,
            losses,
            prompt=self.encoder_l2.weight,
            num_samples=self.num_p_samples,
            held_out_half=self.held_out_prompt_ranking,
        )
        if self.prompt_memory:
            p_tilde_2 = np.concatenate([p_tilde_2, self.get_from_memory(1)], 0)

        # sum over all samples
        # build array: (num_samples, num_p_samples)
        evals = []
        for i in range(x.shape[0]):
            for k in range(p_tilde_2.shape[0]):
                evals.append((x[i], y[i], p_tilde_2[k]))

        if self.output_scoring_function == "logprobs":
            # batch_size, num_p_samples
            ll = self.encoder_l2.log_p(
                inputs=np.array([eval[0] for eval in evals]),
                targets=np.array([eval[1] for eval in evals]),
                prompts=np.array([eval[2] for eval in evals]),
                output_classes=self.output_classes,
                agg="sum" if self.forward_use_classes else "max",
            ).logp_targets

            # batch_size, num_p_samples
            ll = ll.reshape(batch_size, p_tilde_2.shape[0])

            p2_elbo = ll.mean(axis=0)
            self.result_entry.log_candidates(p_tilde_2, p2_elbo)
            best_p2 = p_tilde_2[np.argmax(p2_elbo)]
            best_p2_elbo = np.max(p2_elbo)

            log_message("--- P2 ---")
            for i, (p_tilde_2_i, p2_elbo_i) in enumerate(zip(p_tilde_2, p2_elbo)):
                log_message("#", i, "ELBO", p2_elbo_i, ",", p_tilde_2_i)
            log_message("----------")

            log_message("Best P2 Index: ", np.argmax(p2_elbo))
            log_message("Best P2: ", best_p2)
            log_message("Best P2 ELBO: ", best_p2_elbo)

            return best_p2_elbo, None, best_p2
        elif self.output_scoring_function == "accuracy":
            if np.sum(losses) == 0.0:
                return 1.0, None, self.encoder_l2.weight

            acc = self.encoder_l2.accuracy(
                inputs=np.array([eval[0] for eval in evals]),
                targets=np.array([eval[1] for eval in evals]),
                prompts=np.array([eval[2] for eval in evals]),
                num_samples=self.num_acc_mc_samples,
                max_tokens=10,
                postprocess_prediction=postprocess_prediction,
            )
            acc = acc.reshape(batch_size, p_tilde_2.shape[0]).mean(0)

            best_p2_idx = np.argmax(acc)
            best_p2 = p_tilde_2[best_p2_idx]
            best_p2_acc = np.max(acc)

            log_message("--- P2 ---")
            for i, (p_tilde_2_i, p2_acc_i) in enumerate(zip(p_tilde_2, acc)):
                log_message("#", i, "ACC", p2_acc_i, ",", p_tilde_2_i)
            log_message("----------")

            log_message("Best P2 Index: ", best_p2_idx)
            log_message("Best P2: ", best_p2)
            log_message("Best P2 ACC: ", best_p2_acc)
            return best_p2_acc, None, best_p2
        else:
            raise NotImplementedError()

    def sample_hidden_states(
        self,
        x,
        y,
        h1,
        include_h1=False,
    ):
        # samples from the approx. posterior of h_1
        # (batch_size, num_h_samples)
        # q(h | x, y, p_1, p_2)
        batch_size = x.shape[0]

        if not self.num_h_samples and not include_h1:
            raise ValueError("Must sample at least one h or include h1")

        if self.num_h_samples:
            h_tilde_1 = self.q_sampler.sample_q_h(
                x=x,
                y=y,
                h=h1,
                prompt=self.encoder_l1.weight,
                next_prompt=self.encoder_l2.weight,
                num_samples=self.num_h_samples,
                return_logprobs=False,
            )

        # concatenate the original sample
        if include_h1:
            log_message(
                colored("Concatenating original sample to h_tilde_1!", "yellow")
            )
            if self.num_h_samples:
                h_tilde_1 = np.concatenate([h1[:, None], h_tilde_1], axis=1)
            else:
                h_tilde_1 = h1[:, None]

        num_h_samples = h_tilde_1.shape[1]

        ## TIGHTEN POSTERIOR APPROXIMATION...
        ## e.g. compute log p(y | ~h, p_2, x) + log p(~h | x, p_1)
        # compute log p(y | ~h, p_2) (residual connection added!)
        x_repeat = x.repeat(num_h_samples, axis=0)
        residual_h_tilde_1 = self.encoder_l1.apply_residual(
            h_tilde_1.flatten(), x_repeat
        ).reshape(batch_size, num_h_samples)

        if num_h_samples > 1 and self.posterior_temp < 100.0:
            log_message(colored("Tightening posterior approximation...", "yellow"))
            y_repeat = y.repeat(num_h_samples, axis=0)

            if self.output_scoring_function == "logprobs":
                ll = self.encoder_l2.log_p(
                    inputs=residual_h_tilde_1.flatten(),
                    targets=y_repeat.flatten(),
                    output_classes=self.output_classes,
                    agg="sum" if self.forward_use_classes else "max",
                ).logp_targets
                logits = ll.reshape(batch_size, num_h_samples)

                if self.posterior_sharpening_include_prior:
                    # now compute the prior log-prob of ~h, log p(~h | x, p_1)
                    log_message(
                        colored(
                            "Scoring posterior samples only with log-likelihood + prior",
                            "yellow",
                        )
                    )

                    # pr
                    pr = self.encoder_l1.log_p(
                        x_repeat, h_tilde_1.flatten()
                    ).logp_targets.reshape(batch_size, num_h_samples)
                    logits = logits + pr
                else:
                    log_message(
                        colored(
                            "Scoring posterior samples only with log-likelihood!",
                            "yellow",
                        )
                    )

                if self.posterior_sharpening_use_mi_regularization:
                    log_message(
                        colored(
                            "Scoring posterior samples with MI regularization!",
                            "yellow",
                        )
                    )

                    # mi regularization term
                    mi = self.encoder_l2.log_p(
                        h_tilde_1.flatten(), y_repeat.flatten()
                    ).logp_targets.reshape(batch_size, num_h_samples)

                    logits = logits - mi

            elif self.output_scoring_function == "accuracy":
                logits = self.encoder_l2.accuracy(
                    inputs=residual_h_tilde_1.flatten(),
                    targets=y_repeat.flatten(),
                    num_samples=self.num_acc_mc_samples,
                    postprocess_prediction=postprocess_prediction,
                ).reshape(batch_size, num_h_samples)
        else:
            logits = np.zeros((batch_size, num_h_samples))

        # posterior weights for h_tilde_1, (batch_size, num_h_samples,)
        weights = np.exp(logits / self.posterior_temp) / np.sum(
            np.exp(logits / self.posterior_temp), axis=1, keepdims=True
        )
        assert (weights.sum(1).sum(0) - batch_size) < 1e-5

        # get best hidden state
        best_h_tilde_1_index: np.array = np.argmax(weights, axis=1)
        residual_h_tilde_1_star = residual_h_tilde_1[
            np.arange(batch_size), best_h_tilde_1_index
        ]
        h_tilde_1_star = h_tilde_1[np.arange(batch_size), best_h_tilde_1_index]
        num_h_samples = h_tilde_1.shape[1]

        log_message("Prior h:", h1[0])
        log_message("Best Posterior h:", h_tilde_1_star[0])
        log_message("Best Posterior index:", best_h_tilde_1_index[0])
        counter = Counter(best_h_tilde_1_index)
        log_message("Best Posterior indices:", counter)

        if self.use_h_argmax:
            h_tilde_1 = h_tilde_1_star[:, None]
            residual_h_tilde_1 = residual_h_tilde_1_star[:, None]
            weights = np.ones((batch_size, 1))

        # return both samples and weights associated with them
        return (
            residual_h_tilde_1,
            h_tilde_1,
            h_tilde_1_star,
            weights,
        )

    def compute_elbo_score(self, log_likes, class_weights=None):
        """
        Args:
            log_likes: (batch_size, num_h_samples, num_p_samples)
        """
        if class_weights is None:
            score = log_likes.mean(0)
        else:
            assert log_likes.shape[1] == class_weights.shape[1]
            score = np.sum(log_likes * class_weights[:, :, None], axis=1).mean(0)
        return score

    def inference_vi(
        self,
        x: np.array,
        h1: np.array,
        r_h1: np.array,
        y: np.array,
        y_hat: np.array,
        losses: np.array,
    ):
        batch_size = x.shape[0]
        assert y.shape[0] == batch_size

        # sample hidden states from the proposal distribution
        (
            residual_h_tilde_1,
            h_tilde_1,
            h_tilde_1_star,
            weights,
        ) = self.sample_hidden_states(
            x=x,
            y=y,
            h1=h1,
            include_h1=False,
        )
        num_h_samples = h_tilde_1.shape[1]

        log_message(colored("Number of h samples: {}".format(num_h_samples), "yellow"))
        log_message(
            colored(
                "Norm. entropy of posterior q(h): {}".format(
                    -(weights * np.log(weights)).sum(-1).mean(0)
                    / (np.log(weights.shape[1]) + 1e-5)
                ),
                "yellow",
            )
        )

        # marginalize over posterior samples
        # build array: (num_samples, num_h_samples, num_p_samples)
        eval_batch_size = batch_size
        eval_x = x
        eval_weights = weights
        eval_r_h_tilde_1 = residual_h_tilde_1
        eval_y = y
        eval_h_tilde_1 = h_tilde_1

        if self.train_p2:
            current_prompt = self.encoder_l2.weight
            p2_elbos = []

            for num_step in range(self.num_p2_steps):
                # sample from the prompt distribution, (num_prompts,)
                p_tilde_2: np.array = self.prompt_sampler_2.sample_q_p(
                    inputs=r_h1,
                    y=y,
                    y_hat=y_hat,
                    losses=losses,
                    prompt=current_prompt,
                    num_samples=self.num_p_samples,
                    held_out_half=self.held_out_prompt_ranking,
                )
                if self.prompt_memory:
                    p_tilde_2 = np.concatenate([p_tilde_2, self.get_from_memory(1)], 0)

                evals = []
                for i in range(eval_batch_size):
                    for j in range(num_h_samples):
                        for k in range(p_tilde_2.shape[0]):
                            evals.append(
                                (
                                    eval_r_h_tilde_1[i, j],
                                    eval_y[i],
                                    p_tilde_2[k],
                                )
                            )

                # batch_size, num_h_samples, num_p_samples
                if self.output_scoring_function == "logprobs":
                    log_message(
                        colored("Evaluating log likelihoods for p2...", "yellow")
                    )

                    scores = self.encoder_l2.log_p(
                        inputs=np.array([eval[0] for eval in evals]),
                        targets=np.array([eval[1] for eval in evals]),
                        prompts=np.array([eval[2] for eval in evals]),
                        output_classes=self.output_classes,
                        agg="sum" if self.forward_use_classes else "max",
                    ).logp_targets
                    scores = scores.reshape(
                        eval_batch_size, num_h_samples, p_tilde_2.shape[0]
                    )

                    # trust factor diminishes changes to output layer
                    if self.trust_factor > 0.0:
                        evals = []
                        for i in range(batch_size):
                            for k in range(p_tilde_2.shape[0]):
                                evals.append((r_h1[i], y[i], p_tilde_2[k]))

                        lps = self.encoder_l2.log_p(
                            inputs=np.array([eval[0] for eval in evals]),
                            targets=np.array([eval[1] for eval in evals]),
                            prompts=np.array([eval[2] for eval in evals]),
                            output_classes=self.output_classes,
                            agg="sum" if self.forward_use_classes else "max",
                        ).distribution
                        lps = lps.reshape(batch_size, p_tilde_2.shape[0], -1)
                        p2_kl = compute_pairwise_kl(lps)
                    else:
                        p2_kl = np.zeros(p_tilde_2.shape[0])

                elif self.output_scoring_function == "accuracy":
                    log_message(colored("Evaluating accuracies for p2...", "yellow"))
                    scores = self.encoder_l2.accuracy(
                        inputs=np.array([eval[0] for eval in evals]),
                        targets=np.array([eval[1] for eval in evals]),
                        prompts=np.array([eval[2] for eval in evals]),
                        num_samples=self.num_acc_mc_samples,
                        postprocess_prediction=postprocess_prediction,
                    )
                    scores = scores.reshape(
                        eval_batch_size, num_h_samples, p_tilde_2.shape[0]
                    )
                    p2_kl = np.zeros(p_tilde_2.shape[0])

                p2_elbo = self.compute_elbo_score(scores, eval_weights)
                p2_reward = p2_elbo - self.trust_factor * p2_kl
                best_p2 = p_tilde_2[np.argmax(p2_reward)]
                best_p2_elbo = np.max(p2_reward)
                best_p2_index = np.argmax(p2_reward)
                current_prompt = best_p2
                p2_elbos.append(best_p2_elbo)

                log_message(
                    f"P2 optimization step done [{num_step + 1}/{self.num_p2_steps}]."
                )
                log_message(f"Optimization metric: {best_p2_elbo}")
                log_message(f"Current prompt selected: {best_p2}")

            log_message("Optimization of P2... DONE.", p2_elbos)
        else:
            p_tilde_2 = np.asarray([self.encoder_l2.weight])
            p2_elbo = np.zeros(self.num_p_samples)
            p2_kl = np.zeros(self.num_p_samples)
            best_p2 = self.encoder_l2.weight
            best_p2_elbo = 0.0
            best_p2_index = 0

        if self.train_p1:
            # update w.r.t. p2 is done at this point, proceed with p1,
            # sample proposals for the first layer prompt given the best ~h, h_tilde_1_star
            current_prompt = self.encoder_l1.weight
            p1_elbos = []

            for num_step in range(self.num_p1_steps):
                p_tilde_1: np.array = self.prompt_sampler_1.sample_q_p(
                    inputs=x,
                    y=h_tilde_1_star,
                    y_hat=h1,
                    losses=losses,
                    prompt=current_prompt,
                    num_samples=self.num_p_samples,
                    held_out_half=self.held_out_prompt_ranking,
                )

                if self.prompt_memory:
                    p_tilde_1 = np.concatenate([p_tilde_1, self.get_from_memory(0)], 0)

                # marginalize over all posterior samples
                # build array: (num_samples, num_h_samples, num_p_samples)
                evals = []
                eval_h_tilde_1_ = np.concatenate([h1[:, None], eval_h_tilde_1], 1)
                scores = self.score_p1(eval_x, eval_h_tilde_1_, p_tilde_1)
                ll_orig = scores[:, 0, :]

                if self.use_nce:
                    weights = np.exp(scores[:, 1:, :]) / np.exp(scores[:, 1:, :]).sum(1)[:, None, :]
                    p1_elbo = (eval_weights[:, :, None] * np.log(weights + 1e-12)).sum(1).mean(0)
                else:
                    p1_elbo = self.compute_elbo_score(scores[:, 1:, :], eval_weights)

                # Compute an exploration like logp penalty that penalizes the log-likelihood of wrong thoughts
                if self.logp_penalty > 0.0:
                    error_terms = np.where(losses > 0.0)[0]

                    if len(error_terms) > 0:
                        ll_errors = ll_orig[error_terms]
                        p1_elbo = (
                            p1_elbo
                            - self.logp_penalty * ll_errors.sum(0) / ll_orig.shape[0]
                        )

                best_p1 = p_tilde_1[np.argmax(p1_elbo)]
                best_p1_elbo = np.max(p1_elbo)
                best_p1_index = np.argmax(p1_elbo)
                current_prompt = best_p1

                p1_elbos.append(best_p1_elbo)

                log_message(
                    f"P1 optimization step done [{num_step + 1}/{self.num_p1_steps}]."
                )
                log_message(f"Optimization metric: {'->'.join(['{:.3f}'.format(e) for e in p1_elbos])}")
                log_message(f"Current prompt selected: {best_p1}")

            log_message("Optimization of P1... DONE.", p1_elbos)
        else:
            p_tilde_1 = np.asarray([self.encoder_l1.weight])
            p1_elbo = np.zeros(self.num_p_samples)
            best_p1 = self.encoder_l1.weight
            best_p1_elbo = 0.0
            best_p1_index = 0

        self.result_entry.log_candidates(p_tilde_2, p2_elbo, p_tilde_1, p1_elbo)
        log_message("--- P1 ---")
        for i, (p_tilde_1_i, p1_elbo_i) in enumerate(zip(p_tilde_1, p1_elbo)):
            log_message("#", i, "ELBO:", p1_elbo_i, ",", p_tilde_1_i)

        log_message("--- P2 ---")
        for i, (p_tilde_2_i, p2_elbo_i, p2_kl_i) in enumerate(
            zip(p_tilde_2, p2_elbo, p2_kl)
        ):
            log_message("#", i, "ELBO:", p2_elbo_i, "XE:", p2_kl_i, ",", p_tilde_2_i)

        log_message("Best P1 Index: ", best_p1_index)
        log_message("Best P2 Index: ", best_p2_index)
        log_message("Best P1: ", best_p1, best_p1_elbo)
        log_message("Best P2: ", best_p2, best_p2_elbo)
        return best_p1_elbo, best_p2_elbo, best_p1, best_p2

    def score_p1(self, eval_x, eval_h_tilde_1, p_tilde_1):
        if self.hidden_scoring_function == "logprobs":
            log_message(colored("Evaluating log likelihoods for p1...", "yellow"))

            evals = []
            for i in range(eval_h_tilde_1.shape[0]):
                for j in range(eval_h_tilde_1.shape[1]):
                    for k in range(p_tilde_1.shape[0]):
                        evals.append(
                            (
                                eval_x[i],
                                eval_h_tilde_1[i, j],
                                p_tilde_1[k],
                            )
                        )
            # (batch_size, num_h_samples, num_p_samples)
            scores = self.encoder_l1.log_p(
                inputs=np.array([eval[0] for eval in evals]),
                targets=np.array([eval[1] for eval in evals]),
                prompts=np.array([eval[2] for eval in evals]),
            ).logp_targets

            scores = scores.reshape(
                eval_h_tilde_1.shape[0],
                eval_h_tilde_1.shape[1],
                p_tilde_1.shape[0],
            )
        else:
            raise NotImplementedError()
        return scores

    def strip_options(self, x):
        """
        In bbh, there is the lame pre-processing that appends the options
        to the input. This function removes them, this can be useful for
        hidden states, where we don't want the model to output the option directly.
        """
        x_ = []
        for x_i in x:
            if "Options:" in x_i:
                x_i = x_i[: x_i.index("Options:")].strip()
            x_.append(x_i)
        return np.array(x_)

    def strip_prefix(self, x):
        """
        Strip prefix from the hidden state if the model generates it.
        """
        x_ = []
        for x_i in x:
            if self.strip_prefix_for_hidden in x_i:
                x_i = x_i[
                    x_i.index(self.strip_prefix_for_hidden)
                    + len(self.strip_prefix_for_hidden) :
                ].strip()
            x_.append(x_i)
        return np.array(x_)

    def forward(self, x, y=None, infos=None, temperature=0.0, cost_only=False):
        """
        Args:
            temperature: temperature to use for the forward pass.
        """
        if self.two_layers:
            if self.strip_options_for_hidden:
                x_stripped = self.strip_options(x)
            else:
                x_stripped = x

            if self.strip_prefix_for_hidden:
                x_stripped = self.strip_prefix(x_stripped)

            h_1_out = self.encoder_l1(
                x_stripped, temperature=temperature, max_tokens=self.p1_max_tokens
            )

            self.cost += self.encoder_l2.forward_evaluate.compute_cost(x_stripped)

            # execute second template
            h_1 = self.encoder_l1.apply_residual(h_1_out, x)
            y_hat = self.encoder_l2(
                h_1,
                output_classes=self.output_classes
                if self.forward_use_classes
                else None,
                temperature=temperature,
                max_tokens=self.p2_max_tokens,
            )

            self.cost += self.encoder_l2.forward_evaluate.compute_cost(h_1)
        else:
            h_1_out, h_1 = None, None

            # format infos (i.e. few-shot examples)
            if infos is not None:
                infos = "\n\n\n".join(
                    [
                        self.encoder_l2.forward_template.render(
                            prompt="", input=info[0], answer=info[1]
                        )
                        for info in infos
                    ]
                )
                x = np.array([infos + "\n\n\n" + x_ for x_ in x])

            x_ = [
                self.encoder_l2.forward_template.render(
                    input=x_, prompt=self.encoder_l2.weight
                )
                for x_ in x
            ]
            self.cost += self.encoder_l2.forward_evaluate.compute_cost(x_)

            # only compute cost! save inference
            if cost_only:
                return ["" for _ in x]

            y_hat = self.encoder_l2(
                x,
                output_classes=self.output_classes
                if self.forward_use_classes
                else None,
                temperature=temperature,
                max_tokens=self.p2_max_tokens,
            )
        self.result_entry.log_hiddens(hiddens=h_1_out, size=len(x))
        self.result_entry.log_outputs(outputs=y_hat)

        y_hat = np.array([self.loss_fn.postproc(y_hat_i) for y_hat_i in y_hat])
        if y is not None:
            y = np.array([self.loss_fn.postproc(y_i) for y_i in y])
            losses = self.loss_fn(y_hat, y)
            if self.two_layers:
                elbo1, elbo2, p1, p2 = self.inference_vi(
                    x_stripped, h_1_out, h_1, y, y_hat, losses
                )
                elbo = elbo1 + elbo2
                return elbo, p1, p2, np.mean(losses), elbo1, elbo2
            else:
                elbo, p1, p2 = self.inference_one_layer(x, y, y_hat, losses)
                return elbo, p1, p2, np.mean(losses), 0.0, elbo
        else:
            return y_hat
