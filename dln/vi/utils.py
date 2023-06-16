import logging

import numpy as np


# Use this function to log messages to the file
def log_message(*messages):
    print(*messages)
    logging.info(" ".join(map(str, messages)))


def compute_pairwise_kl(lps):
    """Compute a TRPO-like trust region constraint for the update.

    We make updates constrained by the KL divergence between the function induced by the current prompt
    and the function induced by the new prompt. This is similar to the TRPO update, but we don't have
    a policy, so we can't compute the KL divergence between policies. Instead, we compute the KL divergence
    between the functions induced by the prompts.
    """
    # compute pairwise kl, considers reference always as the first prompt
    return (
        (lps[:, :1, :] * (np.log(lps[:, :1, :]) - np.log(lps[:, :, :]))).sum(-1).mean(0)
    )
