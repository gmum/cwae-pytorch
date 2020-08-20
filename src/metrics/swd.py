import torch

# Implementation is based on: https://github.com/skolouri/swae/blob/master/MNIST_SlicedWassersteinAutoEncoder_Circle.ipynb
# Code adapted to use PyTorch instead of Tensorflow


def sliced_wasserstein_distance(encoded_samples: torch.Tensor,
                                distribution_samples: torch.Tensor,
                                num_projections: int) -> torch.Tensor:
    assert len(encoded_samples.size()) == 2

    z_dim = encoded_samples.size(1)

    projections = __generate_theta(z_dim, num_projections).type_as(encoded_samples)
    swd = __sliced_wasserstein_distance(encoded_samples, distribution_samples, projections)
    return swd


def __generate_theta(z_dim: int, num_samples: int) -> torch.Tensor:
    return torch.stack([w / torch.sqrt((w**2).sum()) for w in torch.randn([num_samples, z_dim])])


def __sliced_wasserstein_distance(encoded_samples: torch.Tensor,
                                  distribution_samples: torch.Tensor,
                                  projections: torch.Tensor) -> torch.Tensor:
    transposed_projections = projections.transpose(0, 1).detach()
    encoded_projections = encoded_samples.matmul(transposed_projections)
    distribution_projections = distribution_samples.matmul(transposed_projections)
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, 2)
    return wasserstein_distance.mean()
