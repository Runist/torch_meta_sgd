# Pyorch - Meta-SGD

## Part 1. Introduction

As we all know, "Model-Agnostic Meta-Learning" can solve few-shot learning problem. But it only solves the problem of model initialization.

Today, I want to show a new optimizer which call **Meta-SGD**. On the basis of MAML, the accuracy of the model after training can be improved.

If this works for you, please give me a star, this is very important to me.😊

## Part 2. Quick  Start

1. Pull repository.

```shell
git clone https://github.com/Runist/torch_meta_sgd.git
```

2. You need to install some dependency package.

```shell
cd torch_meta_sgd
pip install -r requirements.txt
```

3. Download the *Omiglot* dataset.

```shell
mkdir data
cd data
wget https://github.com/Runist/MAML-keras/releases/download/v1.0/Omniglot.tar
tar -xvf Omniglot.tar
```

4. Start training.

```shell
python train.py
```

```
epoch 1: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.45s/it, loss=1.2326]
=> loss: 1.2917   acc: 0.4990   val_loss: 0.8875   val_acc: 0.7963
epoch 2: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.32s/it, loss=0.9818]
=> loss: 1.0714   acc: 0.6688   val_loss: 0.8573   val_acc: 0.7713
epoch 3: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.34s/it, loss=0.9472]
=> loss: 0.9896   acc: 0.6922   val_loss: 0.8000   val_acc: 0.7773
epoch 4: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.39s/it, loss=0.7929]
=> loss: 0.8258   acc: 0.7812   val_loss: 0.8071   val_acc: 0.7676
epoch 5: 100%|█████████████████████████████████████| 4/4 [00:08<00:00,  2.14s/it, loss=0.6662]
=> loss: 0.7754   acc: 0.7646   val_loss: 0.7144   val_acc: 0.7833
epoch 6: 100%|█████████████████████████████████████| 4/4 [00:04<00:00,  1.21s/it, loss=0.7490]
=> loss: 0.7565   acc: 0.7635   val_loss: 0.6317   val_acc: 0.8130
epoch 7: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.25s/it, loss=0.5380]
=> loss: 0.5871   acc: 0.8333   val_loss: 0.5963   val_acc: 0.8255
epoch 8: 100%|█████████████████████████████████████| 4/4 [00:05<00:00,  1.27s/it, loss=0.5144]
=> loss: 0.5786   acc: 0.8255   val_loss: 0.5652   val_acc: 0.8463
epoch 9: 100%|█████████████████████████████████████| 4/4 [00:04<00:00,  1.18s/it, loss=0.4945]
=> loss: 0.5038   acc: 0.8510   val_loss: 0.6305   val_acc: 0.8005
epoch 10: 100%|█████████████████████████████████████| 4/4 [00:06<00:00,  1.75s/it, loss=0.4634]
=> loss: 0.4523   acc: 0.8719   val_loss: 0.5285   val_acc: 0.8491
```

## Part 3. Train your own dataset
1. You should set same parameters in **args.py**. More detail you can get in my [blog](https://blog.csdn.net/weixin_42392454/article/details/127228377?spm=1001.2014.3001.5501).

```python
parser.add_argument('--train_data_dir', type=str,
                    default="./data/Omniglot/images_background/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_dir', type=str,
                    default="./data/Omniglot/images_evaluation/",
                    help='The directory containing the validation image data.')
parser.add_argument('--n_way', type=int, default=10,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=1,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=1,
                    help='The number of query set image for every task.')
```

2. Start training.

```shell
python train.py --n_way=5 --k_shot=1 --q_query=1
```

## Part 4. Paper and other implement

- [Runist](https://github.com/Runist)/[torch_maml](https://github.com/Runist/torch_maml)
- [Runist](https://github.com/Runist)/[MAML-keras](https://github.com/Runist/MAML-keras)
