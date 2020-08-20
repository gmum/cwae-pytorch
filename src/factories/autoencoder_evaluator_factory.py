from noise_creator import NoiseCreator

from evaluators.sample_evaluators.cw_sample_evaluator import CWSampleEvaluator
from evaluators.sample_evaluators.mardia_kurtosis_sample_evaluator import MardiaKurtosisSampleEvaluator
from evaluators.sample_evaluators.mardia_skewness_sample_evaluator import MardiaSkewnessSampleEvaluator
from evaluators.sample_evaluators.mmd_sample_evaluator import MMDSampleEvaluator
from evaluators.sample_evaluators.swd_sample_evaluator import SWDSampleEvaluator

from evaluators.rec_err_evaluator import RecErrEvaluator

from evaluators.autoencoder_evaluator import AutoEncoderEvaluator


def create_evaluator(noise_creator: NoiseCreator) -> AutoEncoderEvaluator:

    rec_err_evaluator = RecErrEvaluator()

    latent_evaluators = list()
    latent_evaluators.append(('cw', CWSampleEvaluator()))
    latent_evaluators.append(('mmd', MMDSampleEvaluator(noise_creator)))
    latent_evaluators.append(('swd', SWDSampleEvaluator(noise_creator)))
    latent_evaluators.append(('skewness', MardiaSkewnessSampleEvaluator()))
    latent_evaluators.append(('kurtosis', MardiaKurtosisSampleEvaluator()))

    autoencoder_evaluator = AutoEncoderEvaluator(rec_err_evaluator, latent_evaluators)

    return autoencoder_evaluator
