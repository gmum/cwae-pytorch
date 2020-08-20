import torch
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator


class MardiaSkewnessSampleEvaluator(BaseSampleEvaluator):

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        N = sample.size(0)
        skewness = ((1/N**2)*torch.sum(torch.matmul(sample, sample.T)**3))
        return skewness
