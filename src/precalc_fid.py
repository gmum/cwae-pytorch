import argparse
import torch
import numpy as np
from externals.inception import InceptionV3
from factories.dataset_factory import get_dataset
from externals.fid_score import get_predictions_for_batch, calculate_statistics_for_activations


def get_activations_for_dataloader(model: InceptionV3, dataloader: torch.utils.data.DataLoader, cuda: bool, verbose: bool):
    model.eval()

    pred_arr = list()
    for i, data in enumerate(dataloader, 0):
        if verbose:
            print(f'\rPropagating batch {(i + 1)}', end='', flush=True)

        images = data[0]
        if cuda:
            images = images.cuda()

        if images.size(1) == 1:
            images = torch.cat([images for i in range(3)], 1)

        pred = get_predictions_for_batch(images, model, images.size(0))
        pred_arr.append(pred)

    result = np.concatenate(pred_arr)

    if verbose:
        print('done. Result size: ', result.size)

    return result


def calculate_activation_statistics_for_dataloader(model: InceptionV3, dataloader: torch.utils.data.DataLoader, cuda: bool, verbose: bool):
    act = get_activations_for_dataloader(model, dataloader, cuda, verbose)
    return calculate_statistics_for_activations(act)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | cifar10 | celeba')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')

    parser.add_argument('--batch_size', type=int, default=100, help='batch size of dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--outf', type=str, default=None, help='output file path')

    parser.add_argument('--cpu_only', action='store_true', help='disables use of GPU')
    parser.add_argument('--no_verbose', action='store_true', help='disables progress logging')

    args = parser.parse_args()

    use_cuda = not args.cpu_only
    verbose = not args.no_verbose

    output_file = args.outf if args.outf is None else f'../data/{args.dataset}_fid_stats.npz'

    train_dataset = get_dataset(args.dataset, args.dataroot, True)
    validation_dataset = get_dataset(args.dataset, args.dataroot, False)
    dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])

    if verbose:
        print(f'There are {len(dataset)} elements in the dataset')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    if use_cuda:
        model = model.cuda()

    print("Calculate FID stats..", end=" ", flush=True)
    with torch.no_grad():
        mu, sigma = calculate_activation_statistics_for_dataloader(model, dataloader, cuda=use_cuda, verbose=verbose)

    np.savez_compressed(output_file, mu=mu, sigma=sigma)
    print("Finished")


if __name__ == '__main__':
    run()
