
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class NoiseCreator:

    def __init__(self, latent_size: int):
        self.__distribution = MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))

    def create(self, batch_size: int) -> torch.Tensor:
        return self.__distribution.sample([batch_size])
