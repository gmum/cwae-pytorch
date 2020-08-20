import unittest
import torch
from architectures.mnist import Decoder, Encoder


class TestMnistArchitecture(unittest.TestCase):

    def test_decoder_dimmensions(self):
        # Arrange
        latent_dim = 8
        batch_size = 128
        test_input = torch.randn(batch_size, latent_dim)
        decoder = Decoder(latent_dim)

        # Act
        result = decoder(test_input)

        # Assert
        self.assertEqual(4, len(result.size()))
        self.assertEqual(batch_size, result.size(0))
        self.assertEqual(1, result.size(1))
        self.assertEqual(28, result.size(2))
        self.assertEqual(28, result.size(3))

    def test_encoder_dimmensions(self):
        # Arrange
        latent_dim = 64
        batch_size = 128
        test_input = torch.randn(batch_size, 1, 28, 28)
        encoder = Encoder(latent_dim)

        # Act
        result = encoder(test_input)

        # Assert
        self.assertEqual(2, len(result.size()))
        self.assertEqual(batch_size, result.size(0))
        self.assertEqual(latent_dim, result.size(1))
