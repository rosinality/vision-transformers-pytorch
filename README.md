# vision-transformers-pytorch
Implementation of various Vision Transformers (and other vision models) I found interesting

## Models

Currently I have implemented:

### NFNet (https://arxiv.org/abs/2102.06171)

Tested and got 83.17 top-1 accuracy with NFNet-F0

### Pyramid Vision Transformer (https://arxiv.org/abs/2102.12122)

Tested and got 78.94 top-1 accuracy with PVT-Small

### Swin Transformer (https://arxiv.org/abs/2103.14030)

Currently testing Swin-S

### Halo Transformer (https://arxiv.org/abs/2103.12731)

### EfficientNetV2 (https://arxiv.org/abs/2104.00298)

Currently testing EfficienetV2-S. Seems promising, especially progressive adaptive regularization.

## Usage

I'm currently using LMDB of ILSVRC 2012 dataset, that made by using

```bash
python preprocess.py [IMAGENET_PATH] [train/val]
```

I think just using `torchvision.datasets` will be better. I will change to it later.

Then you can do training.

```bash
python train.py --conf [CONFIG FILE] --n_gpu [NUMBER OF GPUS] [Config overrides in form of key=value]
```