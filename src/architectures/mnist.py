import torch
import torch.nn as nn
from modules.view import View


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor):
        assert input_images.size(1) == 1 and input_images.size(2) == 28 and input_images.size(3) == 28
        encoded_latent = self.main(input_images)
        return encoded_latent


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 28 * 28),
            nn.Sigmoid(),
            View(-1, 1, 28, 28)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor):
        decoded_images = self.main(input_latent)
        assert decoded_images.size(1) == 1 and decoded_images.size(2) == 28 and decoded_images.size(3) == 28
        return decoded_images
