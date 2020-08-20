from architectures.mnist import Encoder as MnistEncoder, Decoder as MnistDecoder
from architectures.cifar10 import Encoder as Cifar10Encoder, Decoder as Cifar10Decoder
from architectures.celeba import Encoder as CelebaEncoder, Decoder as CelebaDecoder


def get_architecture(identifier: str, z_dim: int):
    if identifier == 'mnist' or identifier == 'fmnist':
        return MnistEncoder(z_dim), MnistDecoder(z_dim)

    if identifier == 'cifar10':
        return Cifar10Encoder(z_dim), Cifar10Decoder(z_dim)

    if identifier == 'celeba':
        return CelebaEncoder(z_dim), CelebaDecoder(z_dim)

    raise ValueError("Unknown architecture")
