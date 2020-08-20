import unittest
import torch
from architectures.celeba import Decoder, Encoder


class TestCelebaArchitecture(unittest.TestCase):

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
        self.assertEqual(64, result.size(2))
        self.assertEqual(64, result.size(3))

    def test_encoder_dimmensions(self):
        # Arrange
        latent_dim = 64
        batch_size = 128
        test_input = torch.randn(batch_size, 3, 64, 64)
        encoder = Encoder(latent_dim)

        # Act
        result = encoder(test_input)

        # Assert
        self.assertEqual(2, len(result.size()))
        self.assertEqual(batch_size, result.size(0))
        self.assertEqual(latent_dim, result.size(1))
