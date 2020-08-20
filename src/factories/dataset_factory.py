from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA


def get_dataset(identifier: str, dataroot: str, train: bool):
    resolvers = {
        'fmnist': get_fmnist_dataset,
        'mnist': get_mnist_dataset,
        'cifar10': get_cifar10_dataset,
        'celeba': get_celeba_dataset
    }
    return resolvers[identifier](dataroot, train)


def get_mnist_dataset(dataroot: str, train: bool):
    dataset_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return MNIST(root=dataroot,
                 train=train,
                 download=True,
                 transform=dataset_transforms)


def get_fmnist_dataset(dataroot: str, train: bool):
    dataset_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return FashionMNIST(root=dataroot,
                        train=train,
                        download=True,
                        transform=dataset_transforms)


def get_cifar10_dataset(dataroot: str, train: bool):
    dataset_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    return CIFAR10(root=dataroot,
                   train=train,
                   download=True,
                   transform=dataset_transforms)


def get_celeba_dataset(dataroot: str, train: bool):
    celeba_transforms = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    split = 'train' if train else 'valid'
    return CelebA(root=dataroot,
                  split=split,
                  download=True,
                  transform=celeba_transforms)
