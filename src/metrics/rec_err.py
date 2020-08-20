import torch


def per_image_se(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size() == y.size()
    return ((x-y)**2).sum(list(range(1, len(x.size()))))


def mean_per_image_se(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size() == y.size()
    return per_image_se(x, y).mean()
