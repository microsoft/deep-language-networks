from abc import ABC

import numpy as np


class LLoss(ABC):
    pass


class ZeroOneLoss(LLoss):
    def __init__(self, postproc=None):
        """
        Args:
            postproc: a function that takes and returns a string to be apply before calculating the loss
        Returns:
            ZeroOneLoss as an float32 np.array
        """
        self._postproc = postproc

    def __call__(self, input, target):
        """
        Args:
            input: a list of strings
            target: a list of strings
        """
        if self._postproc:
            input = [self.postproc(i) for i in input]
            target = [self.postproc(t) for t in target]
        losses = (np.array(input) != np.array(target)).astype("float32")
        return losses

    @property
    def postproc(self):
        if self._postproc is None:
            return lambda x: x
        return self._postproc