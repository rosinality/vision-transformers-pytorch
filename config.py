from typing import Union, Optional, Tuple

from tensorfn.config import (
    get_model,
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    Checker,
)
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool


SwinTransformer = get_model("swin_transformer")
EfficientNet = get_model("efficientnet")
EfficientNetV2 = get_model("efficientnetv2")

Arch = Union[SwinTransformer, EfficientNet, EfficientNetV2]


class Progressive(Config):
    step: StrictInt = 0
    train_sizes: Tuple[StrictInt, StrictInt] = (128, 300)
    valid_sizes: Tuple[StrictInt, StrictInt] = (128, 300)
    randaug_layers: Tuple[StrictInt, StrictInt] = (2, 2)
    randaug_magnitudes: Tuple[StrictFloat, StrictFloat] = (5, 15)
    mixups: Tuple[StrictFloat, StrictFloat] = (0, 0)
    cutmixes: Tuple[StrictFloat, StrictFloat] = (0, 1)
    dropouts: Tuple[StrictFloat, StrictFloat] = (0.1, 0.3)
    drop_paths: Tuple[StrictFloat, StrictFloat] = (0.2, 0.2)
    verbose: StrictBool = True


class Training(Config):
    optimizer: Optimizer
    scheduler: Scheduler

    base_lr: StrictFloat
    weight_decay: StrictFloat
    wd_skip: StrictStr
    epoch: StrictInt
    ema: StrictFloat
    ema_bn: StrictBool
    agc: StrictFloat
    train_size: StrictInt
    valid_size: StrictInt
    randaug_layer: StrictInt
    randaug_magnitude: StrictFloat
    randaug_increasing: StrictBool = False
    randaug_magnitude_std: StrictFloat = 0.0
    mixup: StrictFloat
    cutmix: StrictFloat
    clip_grad_norm: StrictFloat = 0.0

    progressive: Progressive

    dataloader: DataLoader


class ImageNetConfig(MainConfig):
    arch: Arch
    training: Training
    dataset_path: StrictStr
    log_freq: StrictInt = 10
    checker: Checker
