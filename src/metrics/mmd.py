import torch
from noise_creator import NoiseCreator

# Implementation is based on: https://github.com/tolstikhin/wae/blob/master/wae.py
# Code adapted to use PyTorch instead of Tensorflow


def mmd_penalty(sample_qz: torch.Tensor, sample_pz: torch.Tensor):
    assert len(sample_qz.size()) == 2

    N, D = sample_pz.size()

    nf = float(N)

    sigma2_p = 1. ** 2
    norms_pz = (sample_pz**2).sum(1, keepdim=True)
    distances_pz = norms_pz + norms_pz.t() - 2. * torch.mm(sample_pz, sample_pz.t())

    norms_qz = (sample_qz**2).sum(1, keepdim=True)
    distances_qz = norms_qz + norms_qz.t() - 2. * torch.mm(sample_qz, sample_qz.t())

    dotprods = torch.mm(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprods

    Cbase = 2. * D * sigma2_p
    Cbase = Cbase
    stat = 0.
    TempSubtract = 1. - torch.eye(N).type_as(sample_qz)
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz) + C / (C + distances_pz)
        res1 = torch.mul(res1, TempSubtract)
        res1 = res1.sum() / (nf * nf - nf)
        res2 = C / (C + distances)
        res2 = res2.sum() * 2. / (nf * nf)
        stat += res1 - res2
    return stat
