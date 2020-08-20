import torch
from metrics.cw import cw_choose
from evaluators.sample_evaluators.base_sample_evaluator import BaseSampleEvaluator


class CWSampleEvaluator(BaseSampleEvaluator):

    def evaluate(self, sample: torch.Tensor) -> torch.Tensor:
        z_dim = sample.size(1)
        cw_metric = cw_choose(z_dim)
        return cw_metric(sample)
