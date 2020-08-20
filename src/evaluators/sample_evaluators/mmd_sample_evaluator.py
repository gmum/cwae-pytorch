import torch
from metrics.mmd import mmd_penalty
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator
from noise_creator import NoiseCreator


class MMDSampleEvaluator(BaseSampleEvaluator):

    def __init__(self, noise_creator: NoiseCreator):
        self.__noise_creator = noise_creator

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        comparision_sample = self.__noise_creator.create(sample.size(0)).type_as(sample)
        mmd_penalty_value = mmd_penalty(sample, comparision_sample)
        return mmd_penalty_value
