# Repository info

This repository contains an implementation of Cramer-Wold AutoEncoder(CWAE) in PyTorch, proposed by [Szymon Knop, Jacek Tabor, Przemysław Spurek, Igor Podolak, Marcin Mazur, Stanisław Jastrzębski (2018)](https://arxiv.org/abs/1805.09235).

# Contents of the repository

```
|-- src/ - contains an implementation of CWAE allowing to reproduce experiments from the original paper
|---- architectures/ - files containing architectures proposed in the paper
|---- externals/ - code adapted from the [external repository](https://github.com/mseitzer/pytorch-fid) to compute FID Score of models
|---- evaluators/ - implementation of evaluators of metrics that will be reported in experiments
|---- factories/ - factories used to create objects proper objects base on command line arguments
|---- metrics/ - directory containing the implementation of all of the metrics used in paper
|------ cw.py - implementation of various versions CW Distance
|---- modules/ - custom neural network layers used in models
|---- tests/ - a bunch of unit tests
|---- train_autoencoder.py - the main script to run all of the experiments
|---- precalc_fid.py - additional script that can be used to precalculate FID statistics for datasets
|-- results/ - directory that will be created to store the results of conducted experiments
|-- data/ - default directory that will be used as a source of data and place to download datasets
```

Experiments are written in `pytorch-lightning` to decouple the science code from the engineering. The `LightningModule` implementation is in `train_autoencoder.py` file. For more details refer to [PyTorch-Lightning documentation](https://github.com/PyTorchLightning/pytorch-lightning)

# Conducting the experiments

To execute experiments described in Table 1 in the paper run scripts located in `src/reproduce_table1.sh`

Apart from CWAE model, the repository supports running WAE, SWAE, and vanilla AE models. All of the implementations are based on the respective papers and repositories.

- For Wasserstein AutoEncoders [arXiv](https://arxiv.org/abs/1711.01558) and [GitHub repository](https://github.com/tolstikhin/wae)

- For Sliced-Wasserstein AutoEncoders [arXiv](https://arxiv.org/pdf/1804.01947.pdf) and [GitHub repository](https://github.com/skolouri/swae)

## Browsing the results

Results are stored in tensorboard format. To browse them run the following command:
`tensorboard --logdir results`

## Reproducing the results

Below are the obtained FID scores for experiments conducted with this repository's code:
|              	|  SWAE 	| WAE-MMD 	|  CWAE 	|
|--------------	|------:	|--------:	|------:	|
| MNIST        	| 30.94 	|  28.71  	| 24.22 	|
| FashionMNIST 	| 55.99 	|  51.74  	| 50.35 	|
| CIFAR-10     	| 131.6 	|  136.6  	| 118.1 	|
| CELEBA       	| 62.42 	|  51.29  	| 48.02 	|

Reported results may vary a little from the ones reported in the paper because implemented architectures in PyTorch slightly differ from the original ones. Relations between FID Scores obtained by different models are the same as the ones reported in the paper.

## Other options

The code allows manipulating some of the parameters(for example using other versions of the model, changing learning rate values) for more info see the list of available arguments in `src/args_parser.py` file

To run the unit tests execute the following command:
`python -m unittest`

# Datasets

The repository uses default datasets provided by PyTorch for MNIST, FashionMNIST, CIFAR-10 and CELEBA. To convert CELEB-A to 64x64 images we first center crop images to 140x140 and then resize them to 64x64.

# Environment

- python3
- pytorch
- torchvision
- numpy
- pytorch-lightning

# Additional links

To compute FID Scores we have adapted the code from:

- https://github.com/mseitzer/pytorch-fid

Commit: 011829daeccc84341c1e8e6061d10a640a495573)\*

We based this repository by original TensorFlow CWAE implementation

- https://github.com/gmum/cwae

# License

This implementation is licensed under the MIT License
