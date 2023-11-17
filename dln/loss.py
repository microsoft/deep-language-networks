from abc import ABC, abstractmethod
import re
from typing import Callable, Iterable, Optional, Union

import numpy as np


class LossRegistry:

    _available_losses = {}

    @classmethod
    def register(cls, loss_type: str):
        def inner(lloss):
            cls._available_losses[loss_type] = lloss
            return lloss
        return inner

    @classmethod
    def available_losses(cls):
        return list(cls._available_losses.keys())

    @classmethod
    def instantiate(cls, loss_type: str, postproc: Optional[Callable] = None) -> "LLoss":
        try:
            return cls._available_losses[loss_type](postproc)
        except KeyError:
            raise ValueError(f'Unknown loss type: {loss_type}')


class LLoss(ABC):

    def __init__(self, postproc: Optional[Callable] = None):
        """
        Args:
            postproc: a function that takes and returns a string to be apply before calculating the loss
        """
        self._postproc = postproc

    @property
    def postproc(self):
        """
        Returns the post-processing function for the loss function.
        If the post-processing function has not been set, returns the identity function
        """
        if self._postproc is None:
            return lambda x: x
        return self._postproc

    @abstractmethod
    def loss(self, inputs: Iterable[str], targets: Iterable[str]) -> np.array:
        """
        Computes the loss between the input and target
        Args:
            input: The predicted outputs
            target: The true outputs
        Returns:
            The computed loss
        """
        pass

    def __call__(
        self,
        inputs: Union[str, Iterable[str]],
        targets: Union[str, Iterable[str]],
    ) -> np.array:
        """
        Calls the loss function. If inputs or targets are not iterables, they are converted to lists
        Args:
            inputs: The predicted outputs
            targets: The true outputs
        Returns:
            The computed loss as an np.array
        """
        if isinstance(inputs, str) or not isinstance(inputs, Iterable):
            inputs = [inputs]
        if isinstance(targets, str) or not isinstance(targets, Iterable):
            targets = [targets]
        if self._postproc:
            inputs = [self.postproc(i) for i in inputs]
            targets = [self.postproc(t) for t in targets]
        losses = self.loss(inputs, targets)
        return losses


@LossRegistry.register("exact_match_loss")
class ExactMatchLoss(LLoss):
    """
    Calculates the exact match loss between the predicted and target outputs,
    where 0 indicates a correct prediction and 1 indicates an incorrect prediction.
    """
    def loss(self, inputs: Iterable[str], targets: Iterable[str]) -> np.array:
        losses = (np.array(inputs) != np.array(targets)).astype("float32")
        return losses


@LossRegistry.register("number_presence_loss")
class NumberPresenceLoss(LLoss):
    """
    Calculates the loss based on the presence of a number in a string.
    0 if the target number is present in the input, 1 otherwise.
    """
    def loss(self, inputs: Iterable[str], targets: Iterable[str]) -> np.array:
        losses = []
        for i, t in zip(inputs, targets):
            # Convert the target to float
            number = float(str(t).replace(",", ""))
            # Extract all numbers from the input. dln.Value requires a conversion to str
            numbers_in_text = re.findall(r'\b\d*\.?\d+\b', str(i).replace(",", ""))
            # Try to convert each extracted string into a float and compare it to the number
            # we can match just the last number if we consider that the last number is the answer
            _loss = 1
            for num_str in numbers_in_text:
                if float(num_str) == number:
                    _loss = 0
            losses.append(_loss)
        return np.array(losses).astype("float32")
