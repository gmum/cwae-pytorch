import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from args_parser import parse_args
from evaluators.autoencoder_evaluator import AutoEncoderEvaluator
from evaluators.fid_evaluator import FidEvaluator
import pytorch_lightning as pl
import itertools
from torch.utils.data.dataloader import DataLoader
from factories.cost_function_factory import get_cost_function
from factories.architecture_factory import get_architecture
from factories.optimizer_factory import get_optimizer_factory
from factories.dataset_factory import get_dataset
from factories.autoencoder_evaluator_factory import create_evaluator
from factories.fid_evaluator_factory import create_fid_evaluator
from numpy.random import randint
from noise_creator import NoiseCreator


class AutoEncoderModule(pl.LightningModule):

    def __init__(self, noise_creator: NoiseCreator, autoencoder_evaluator: AutoEncoderEvaluator, fid_evaluator: FidEvaluator, args):
        super().__init__()
        self.__encoder, self.__decoder = get_architecture(args.dataset, args.z_dim)
        self.__noise_creator = noise_creator
        self.__autoencoder_evaluator = autoencoder_evaluator
        self.__fid_evaluator = fid_evaluator
        self.__args = args
        self.__cost_function = get_cost_function(args.model, args.lambda_val, args.z_dim, noise_creator)
        self.__validation_dataloader: DataLoader = None
        self.__test_reconstruction_images: torch.Tensor = None
        self.__random_sampled_latent: torch.Tensor = None

        self.hparams = {
            'lr': args.lr,
            'optimizer': args.optimizer,
            'lambda_val': args.lambda_val,
            'batch_size': args.batch_size
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_factory(self.parameters(), self.__args.lr)

    def train_dataloader(self) -> DataLoader:
        dataset = get_dataset(self.__args.dataset, self.__args.dataroot, train=True)

        return DataLoader(dataset,
                          batch_size=self.__args.batch_size,
                          shuffle=True,
                          num_workers=int(self.__args.workers))

    def val_dataloader(self) -> DataLoader:
        dataset = get_dataset(self.__args.dataset, self.__args.dataroot, train=False)

        self.__test_reconstruction_images = torch.stack([dataset[i][0] for i in randint(0, len(dataset), size=32)])
        assert len(self.__test_reconstruction_images.size()) == 4 and self.__test_reconstruction_images.size(0) == 32

        self.__random_sampled_latent = self.__noise_creator.create(64)

        return DataLoader(dataset, batch_size=self.__args.batch_size, shuffle=False, num_workers=1, drop_last=True)

    def forward(self, batch: torch.Tensor) -> tuple:
        latent = self.__encoder(batch)
        output_images = self.__decoder(latent)
        return latent, output_images

    def get_generator(self) -> torch.nn.Module:
        return self.__decoder

    def training_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]
        latent, output_images = self(batch_images)
        loss = self.__cost_function(batch_images, latent, output_images)

        return {
            'loss': loss,
            'log': {'loss': loss}
        }

    def validation_step(self, batch: torch.Tensor, _) -> dict:
        batch_images = batch[0]

        latent, output_images = self(batch_images)
        loss = self.__cost_function(batch_images, latent, output_images)

        evaluation_metrics = self.__autoencoder_evaluator.evaluate(batch_images, latent, output_images)
        evaluation_metrics['val_loss'] = loss

        return evaluation_metrics

    def validation_epoch_end(self, outputs: dict) -> dict:

        _, decoded_images = self(self.__test_reconstruction_images.to(self.device))

        reconstructions = torch.stack(list(itertools.chain(*zip(self.__test_reconstruction_images.to(self.device), decoded_images))))
        sampled_images = self.__decoder(self.__random_sampled_latent.to(self.device))

        self.logger.experiment.add_images('sampled_images', sampled_images, self.current_epoch)
        self.logger.experiment.add_images('reconstructions', reconstructions, self.current_epoch)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        if self.__fid_evaluator is not None:
            fid_score = self.__fid_evaluator.evaluate(self.get_generator(), self.device)
            tensorboard_logs['fid_score'] = fid_score

        for k in outputs[0].keys():
            tensorboard_logs[k] = torch.stack([x[k] for x in outputs]).mean()

        print(tensorboard_logs)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


def run():

    args = parse_args()

    print(f'Using random seed: {pl.seed_everything(args.random_seed)}')

    output_dir = f'../results/{args.dataset}/{args.model}'
    os.makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    noise_creator = NoiseCreator(args.z_dim)
    autoencoder_evaluator = create_evaluator(noise_creator)
    fid_evaluator = create_fid_evaluator(args.eval_fid, noise_creator) if args.eval_fid is not None else None
    autoencoder_model = AutoEncoderModule(noise_creator, autoencoder_evaluator, fid_evaluator, args)

    trainer = pl.Trainer(gpus=args.ngpu,
                         max_epochs=args.max_epochs_count,
                         progress_bar_refresh_rate=20,
                         check_val_every_n_epoch=args.report_after_epoch,
                         default_root_dir=output_dir)
    trainer.fit(autoencoder_model)


if __name__ == '__main__':
    run()
