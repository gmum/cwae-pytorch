import abc
import torch
import numpy as np


class BaseSampleEvaluator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        pass
