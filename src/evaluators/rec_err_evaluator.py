import torch
from metrics.rec_err import mean_per_image_se


class RecErrEvaluator:

    def evaluate(self, input_images: torch.Tensor, output_images: torch.Tensor) -> torch.Tensor:
        assert input_images.size() == output_images.size()

        flattened_input_images = torch.flatten(input_images, start_dim=1)
        flattened_output_images = torch.flatten(output_images, start_dim=1)

        return mean_per_image_se(flattened_input_images, flattened_output_images)
