import torch
import unittest
from metrics.rec_err import per_image_se, mean_per_image_se


class TestRecErr(unittest.TestCase):

    def test_per_image_se(self):
        # Arrange
        input_images = torch.zeros(3, 3, 64, 64)
        output_images = torch.ones(3, 3, 64, 64)

        output_images[1] = output_images[1] * 2
        output_images[2] = output_images[2] * 3

        # Act
        result = per_image_se(input_images, output_images)

        # Assert
        self.assertEqual(64*64*3, result[0])
        self.assertEqual(64*64*3*4, result[1])
        self.assertEqual(64*64*3*9, result[2])

    def test_per_image_se_different_values_in_channels(self):
        # Arrange
        input_images = torch.zeros(1, 3, 64, 64)
        output_images = torch.ones(1, 3, 64, 64)

        output_images[0][1] = output_images[0][1] * 2
        output_images[0][2] = output_images[0][2] * 3

        # Act
        result = per_image_se(input_images, output_images)

        # Assert
        self.assertEqual(64*64*(1+4+9), result[0])

    def test_mean_per_image_se(self):
        # Arrange
        input_images = torch.zeros(3, 3, 64, 64)
        output_images = torch.ones(3, 3, 64, 64)

        output_images[1] = output_images[1] * 2
        output_images[2] = output_images[2] * 3

        # Act
        result = mean_per_image_se(input_images, output_images)

        # Assert
        self.assertEqual(64*64*3*(1 + 4 + 9) / 3, result)

    def test_mean_per_image_se_2d(self):
        # Arrange
        input_images = torch.zeros(3, 64)
        output_images = torch.ones(3, 64)

        output_images[1] = output_images[1] * 2
        output_images[2] = output_images[2] * 3

        # Act
        result = mean_per_image_se(input_images, output_images)

        # Assert
        self.assertEqual(64*(1 + 4 + 9) / 3, result)
