# vision-transformers-pytorch
Implementation of various Vision Transformers (and other vision models) I found interesting

## Models

Currently I have implemented:

### ViT (https://arxiv.org/abs/2010.11929)

Implemented.

### DINO (https://arxiv.org/abs/2104.14294)

Implemented. Currently testing.

### NFNet (https://arxiv.org/abs/2102.06171)

Tested and got 83.17 top-1 accuracy with NFNet-F0

### Pyramid Vision Transformer (https://arxiv.org/abs/2102.12122)

Tested and got 78.94 top-1 accuracy with PVT-Small

### Swin Transformer (https://arxiv.org/abs/2103.14030)

Tested and got 82.192 on top-1. Re-experimenting with random erasing.

### Halo Transformer (https://arxiv.org/abs/2103.12731)

Implemented.

### EfficientNetV2 (https://arxiv.org/abs/2104.00298)

Tested and got 82.862 on top-1 @ 300px, 83.2 on top-1 @ 380px

### Twins-SVT (https://arxiv.org/abs/2104.13840)

Implemented.

## Usage

I'm currently using LMDB of ILSVRC 2012 dataset, that made by using

```bash
python preprocess.py [IMAGENET_PATH] [train/val]
```

I think just using `torchvision.datasets` will be better. I will change to it later.

Then you can do training.

```bash
python train.py --conf [CONFIG FILE] --n_gpu [NUMBER OF GPUS] [Config overrides in the form of key=value]
```