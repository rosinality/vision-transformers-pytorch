from typing import Union, Optional, Tuple, Sequence, List

from tensorfn.config import (
    get_models,
    get_model,
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    checker,
    Checker,
    TypedConfig,
)
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool


# SwinTransformer = get_model("swin_transformer", "model")
# EfficientNet = get_model("efficientnet", "model")
# EfficientNetV2 = get_model("efficientnetv2", "model")
# NFEfficientNetV2 = get_model("nfefficientnetv2", "model")

# Arch = Union[SwinTransformer, EfficientNet, EfficientNetV2, NFEfficientNetV2]
Arch = get_models("model")


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
    grad_accumulation: Optional[List] = None
    verbose: StrictBool = True


class DINO(TypedConfig):
    __type__ = "dino"

    global_crop_size: StrictInt = 224
    local_crop_size: StrictInt = 96
    global_crop_scale: Tuple[StrictFloat, StrictFloat] = (0.4, 1.0)
    local_crop_scale: Tuple[StrictFloat, StrictFloat] = (0.05, 0.4)
    n_local_crop: StrictInt = 8
    student_drop_path: StrictFloat = 0.1
    warmup_teacher_temperature: StrictFloat = 0.04
    teacher_temperature: StrictFloat = 0.07
    warmup_teacher_temperature_epoch: StrictInt = 30
    teacher_momentum: StrictFloat = 0.9
    weight_decay_end: StrictFloat = 0.4
    freeze_last_layer: StrictInt = 1


class Training(Config):
    optimizer: Optimizer
    scheduler: Scheduler

    base_lr: StrictFloat
    weight_decay: StrictFloat
    wd_skip: StrictStr
    epoch: StrictInt
    ema: StrictFloat = 0.0
    ema_bn: StrictBool = False
    agc: StrictFloat = 0.0
    train_size: StrictInt = 224
    valid_size: StrictInt = 224

    randaug_layer: StrictInt = 0
    randaug_magnitude: StrictFloat = 5.0
    randaug_increasing: StrictBool = False
    randaug_magnitude_std: StrictFloat = 0.0
    randaug_cutout: StrictInt = 40

    erasing: StrictFloat = 0.0

    mixup: StrictFloat = 0.0
    cutmix: StrictFloat = 0.0
    mix_before_aug: StrictBool = True

    clip_grad_norm: StrictFloat = 0.0
    grad_accumulation: StrictInt = 1

    progressive: Progressive

    dataloader: DataLoader


class ImageNetConfig(MainConfig):
    arch: Arch
    training: Training
    dataset_path: StrictStr
    task: Optional[Union[DINO]]

    log_freq: StrictInt = 10
    checker: Checker = Checker()
    fp16: StrictBool = False
    logger: StrictStr = "rich"

