import torch
from metrics.swd import sliced_wasserstein_distance
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator
from noise_creator import NoiseCreator


class SWDSampleEvaluator(BaseSampleEvaluator):

    def __init__(self, noise_creator: NoiseCreator):
        self.__noise_creator = noise_creator

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        comparision_sample = self.__noise_creator.create(sample.size(0)).type_as(sample)
        swd_penalty_value = sliced_wasserstein_distance(sample, comparision_sample, 50)
        return swd_penalty_value
