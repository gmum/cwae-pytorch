import unittest
import torch
from architectures.cifar10 import Decoder, Encoder


class TestCifar10Architecture(unittest.TestCase):

    def test_decoder_dimmensions(self):
        # Arrange
        latent_dim = 64
        batch_size = 128
        test_input = torch.randn(batch_size, latent_dim)
        decoder = Decoder(latent_dim)

        # Act
        result = decoder(test_input)

        # Assert
        self.assertEqual(4, len(result.size()))
        self.assertEqual(batch_size, result.size(0))
        self.assertEqual(3, result.size(1))
        self.assertEqual(32, result.size(2))
        self.assertEqual(32, result.size(3))

    def test_encoder_dimmensions(self):
        # Arrange
        latent_dim = 64
        batch_size = 128
        test_input = torch.randn(batch_size, 3, 32, 32)
        encoder = Encoder(latent_dim)

        # Act
        result = encoder(test_input)

        # Assert
        self.assertEqual(2, len(result.size()))
        self.assertEqual(batch_size, result.size(0))
        self.assertEqual(latent_dim, result.size(1))
