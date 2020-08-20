import torch
from evaluators.rec_err_evaluator import RecErrEvaluator


class AutoEncoderEvaluator:

    def __init__(self, rec_err_evaluator: RecErrEvaluator, latent_evaluators: list):
        self.__rec_err_evaluator = rec_err_evaluator
        self.__latent_evaluators = latent_evaluators

    def evaluate(self, input_images: torch.Tensor, latent: torch.Tensor, output_images: torch.Tensor) -> dict:
        rec_err = self.__rec_err_evaluator.evaluate(input_images, output_images)

        epoch_results = {
            'rec_err': rec_err
        }

        for metric_name, latent_evaluator in self.__latent_evaluators:
            metric_value = latent_evaluator.evaluate(latent)
            epoch_results[metric_name] = metric_value

        return epoch_results
