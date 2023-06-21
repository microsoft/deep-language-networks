import logging

import numpy as np


# Use this function to log messages to the file
def log_message(*messages):
    print(*messages)
    logging.info(" ".join(map(str, messages)))


def compute_pairwise_kl(lps):
    """We make updates constrained by the KL divergence between the function induced by the current prompt
    and the function induced by the new prompt. We compute the KL divergence
    between the functions induced by the prompts.

    The distribution induced by the current prompt is the first element of the second axis in lps.
    """
    # compute pairwise kl, considers reference always as the first prompt
    return (
        (lps[:, :1, :] * (np.log(lps[:, :1, :]) - np.log(lps[:, :, :]))).sum(-1).mean(0)
    )
