#Parameters based on table 1 (p. 19) 

python -OO train_autoencoder.py --model ae   --dataset mnist --z_dim 8 --eval_fid ../data/mnist_fid_stats.npz
python -OO train_autoencoder.py --model swae --dataset mnist --z_dim 8 --eval_fid ../data/mnist_fid_stats.npz
python -OO train_autoencoder.py --model wae  --dataset mnist --z_dim 8 --lr 0.0005 --eval_fid ../data/mnist_fid_stats.npz
python -OO train_autoencoder.py --model cwae --dataset mnist --z_dim 8 --eval_fid ../data/mnist_fid_stats.npz

python -OO train_autoencoder.py --model ae   --dataset fmnist --z_dim 8 --eval_fid ../data/fmnist_fid_stats.npz
python -OO train_autoencoder.py --model swae --dataset fmnist --z_dim 8 --lambda_val 100 --eval_fid ../data/fmnist_fid_stats.npz
python -OO train_autoencoder.py --model wae  --dataset fmnist --z_dim 8 --lambda_val 100 --eval_fid ../data/fmnist_fid_stats.npz
python -OO train_autoencoder.py --model cwae --dataset fmnist --z_dim 8 --lambda_val 10  --eval_fid ../data/fmnist_fid_stats.npz

python -OO train_autoencoder.py --model ae   --dataset cifar10 --z_dim 64 --eval_fid ../data/cifar10_fid_stats.npz
python -OO train_autoencoder.py --model swae --dataset cifar10 --z_dim 64 --eval_fid ../data/cifar10_fid_stats.npz
python -OO train_autoencoder.py --model wae  --dataset cifar10 --z_dim 64 --eval_fid ../data/cifar10_fid_stats.npz
python -OO train_autoencoder.py --model cwae --dataset cifar10 --z_dim 64 --eval_fid ../data/cifar10_fid_stats.npz

python -OO train_autoencoder.py --model ae   --dataset celeba --z_dim 64 --report_after_epoch 5 --max_epochs_count 55 --eval_fid ../data/celeba_fid_stats.npz
python -OO train_autoencoder.py --model swae --dataset celeba --z_dim 64 --lambda_val 100 --lr 0.0005 --report_after_epoch 5 --max_epochs_count 55 --eval_fid ../data/celeba_fid_stats.npz
python -OO train_autoencoder.py --model wae  --dataset celeba --z_dim 64 --lambda_val 100 --lr 0.0005 --report_after_epoch 5 --max_epochs_count 55 --eval_fid ../data/celeba_fid_stats.npz
python -OO train_autoencoder.py --model cwae --dataset celeba --z_dim 64 --lambda_val 5   --lr 0.0005 --report_after_epoch 5 --max_epochs_count 55 --eval_fid ../data/celeba_fid_stats.npz
