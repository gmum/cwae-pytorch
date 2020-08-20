import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='cwae | cwae_plus | wae | wae_log | swae | swae_log')
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | cifar10 | celeba')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')

    parser.add_argument('--z_dim', required=True, type=int, help='latent dimension')
    parser.add_argument('--lambda_val', required=False, type=float, default=1.0, help='value of lambda parameter of a cost function')

    parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--max_epochs_count', type=int, default=500, help='max number of epochs to train for')
    parser.add_argument('--report_after_epoch', type=int, default=25, help='number of epochs to train for')

    parser.add_argument('--optimizer', type=str, default='adam', help='selected optimizer, only adam is supported')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizers learning rate value')

    parser.add_argument('--eval_fid', type=str, default=None, help='Reports FID Score if provided path to precalculated stats (npz file)')

    parser.add_argument('--random_seed', type=int, default=None, help='random seed to parameterize')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    opt = parser.parse_args()
    return opt
