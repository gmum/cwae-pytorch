import torch
import torch.nn as nn
from modules.view import View


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.ConstantPad2d((0, 1, 0, 1), 0.0),
            nn.Conv2d(3, 3, 2, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(3, 32, 2, 2, 0),
            nn.ReLU(True),
            nn.ConstantPad2d((0, 1, 0, 1), 0.0),
            nn.Conv2d(32, 32, 2, 1, 0),
            nn.ReLU(True),
            nn.ConstantPad2d((0, 1, 0, 1), 0.0),
            nn.Conv2d(32, 32, 2, 1, 0),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(16*16*32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_size),
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor):
        assert input_images.size(1) == 3 and input_images.size(2) == 32 and input_images.size(3) == 32
        encoded_latent = self.main(input_images)
        assert len(encoded_latent.size()) == 2
        return encoded_latent


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.__sequential_blocks = [
            nn.Linear(latent_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 16*16*32),
            nn.ReLU(True),
            View(-1, 32, 16, 16),
            nn.ConstantPad2d((0, 1, 0, 1), 0.0),
            nn.ConvTranspose2d(32, 32, 2, 1, padding=1),
            nn.ReLU(True),
            nn.ConstantPad2d((0, 1, 0, 1), 0.0),
            nn.ConvTranspose2d(32, 32, 2, 1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 2, 1, 0),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor) -> torch.Tensor:
        assert len(input_latent.size()) == 2
        output_images = self.main(input_latent)
        assert output_images.size(1) == 3 and output_images.size(2) == 32 and output_images.size(3) == 32, output_images.size()
        return output_images
