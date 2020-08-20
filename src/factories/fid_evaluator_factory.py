import numpy as np
from evaluators.fid_evaluator import FidEvaluator, FidComputer

from externals.inception import InceptionV3
from noise_creator import NoiseCreator


def __handle_path_npz(path: str) -> tuple:
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s


def create_fid_evaluator(precalc_path: str, noise_creator: NoiseCreator, batch_size: int = 100, fid_samples_count: int = 10000) -> FidEvaluator:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    precomputed_stats = __handle_path_npz(precalc_path)
    fid_computer = FidComputer(inception_model, precomputed_stats, noise_creator)
    fid_evaluator = FidEvaluator(fid_computer, batch_size, fid_samples_count)
    return fid_evaluator
