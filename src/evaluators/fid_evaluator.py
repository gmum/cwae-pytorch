import torch
import torch.nn
from externals.inception import InceptionV3
from externals.fid_score import calculate_activation_statistics, calculate_frechet_distance
from noise_creator import NoiseCreator
from tqdm import tqdm


class FidComputer:

    def __init__(self, inception_model: InceptionV3, precomputed_stats: tuple, noise_creator: NoiseCreator):
        self.__inception_model = inception_model
        self.__m2, self.__s2 = precomputed_stats
        self.__noise_creator = noise_creator

    def compute(self, generator: torch.nn.Module, device: str, batch_size: int, count: int):
        batches_count = count//batch_size
        sampled_images_list = list()

        inception_model = self.__inception_model.to(device)
        with torch.no_grad():
            for i in tqdm(range(batches_count)):
                input_noise = self.__noise_creator.create(batch_size).to(device)
                sampled_images_now = generator(input_noise)
                if sampled_images_now.size(1) == 1:
                    sampled_images_now = torch.cat([sampled_images_now for i in range(3)], 1)
                sampled_images_list.append(sampled_images_now)
            sampled_images = torch.cat(sampled_images_list, 0)
            return self.__compute_for_images(inception_model, sampled_images)

    def __compute_for_images(self, model: torch.nn.Module, images: torch.Tensor) -> float:
        m1, s1 = calculate_activation_statistics(images, model)
        return calculate_frechet_distance(m1, s1, self.__m2, self.__s2)


class FidEvaluator:

    def __init__(self, fid_computer: FidComputer, batch_size: int = 100, fid_samples_count: int = 10000):
        self.__fid_computer: FidComputer = fid_computer
        self.__batch_size = batch_size
        self.__fid_samples_count = fid_samples_count

    def evaluate(self, generator: torch.nn.Module, device: str):
        fid_score = self.__fid_computer.compute(generator, device,
                                                batch_size=self.__batch_size,
                                                count=self.__fid_samples_count)
        return fid_score
