import torch
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator


class MardiaKurtosisSampleEvaluator(BaseSampleEvaluator):

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        z_dim = sample.size(1)
        kurtosis = torch.mean(sample.norm(dim=1)**4)
        optimal_kurtosis = z_dim * (z_dim + 2)
        normalized_kurtosis = torch.abs(kurtosis - optimal_kurtosis)
        return normalized_kurtosis
