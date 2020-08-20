import torch
from metrics.cw import cw_choose
from metrics.mmd import mmd_penalty
from metrics.rec_err import mean_per_image_se
from metrics.swd import sliced_wasserstein_distance
from noise_creator import NoiseCreator


def get_cost_function(model: str, lambda_val: float, z_dim: int, noise_creator: NoiseCreator):

    def __create_noise_like_sample(sample: torch.Tensor) -> torch.Tensor:
        return noise_creator.create(sample.size(0)).type_as(sample)

    def __mmd_penalty_with_noise(sample: torch.Tensor) -> torch.Tensor:
        return mmd_penalty(sample, __create_noise_like_sample(sample))

    def __swd_with_noise(sample: torch.Tensor) -> torch.Tensor:
        return sliced_wasserstein_distance(sample, __create_noise_like_sample(sample), 50)

    def __recerr_plus_normality_index(normality_metric):
        return lambda x, z, y: mean_per_image_se(x, y) + lambda_val*normality_metric(z)

    def __recerr_plus_normality_index_log(normality_metric):
        return lambda x, z, y: mean_per_image_se(x, y) + lambda_val*torch.log(normality_metric(z) + 1e-7)

    cost_functions = {
        'ae': lambda x, _, y: mean_per_image_se(x, y),
        'cwae': __recerr_plus_normality_index_log(cw_choose(z_dim)),
        'cwae_plus': __recerr_plus_normality_index(cw_choose(z_dim)),
        'wae': __recerr_plus_normality_index(__mmd_penalty_with_noise),
        'wae_log': __recerr_plus_normality_index_log(__mmd_penalty_with_noise),
        'swae': __recerr_plus_normality_index(__swd_with_noise),
        'swae_log': __recerr_plus_normality_index_log(__swd_with_noise),
    }

    return cost_functions[model]
