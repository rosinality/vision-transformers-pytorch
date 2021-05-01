import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorfn import distributed as dist, nsml, get_logger

try:
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali import fn, types, pipeline_def
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

except ImportError:
    pass

from autoaugment import RandAugment
from dataset import LMDBDataset
from mix_dataset import MixDataset
from transforms import RandomErasing


def wd_skip_fn(skip_type):
    def check_wd_skip_fn(name, param):
        if skip_type == "nfnet":
            return "bias" in name or "gain" in name

        elif skip_type == "resnet":
            return "bias" in name or "bn" in name or param.ndim == 1

        elif skip_type == "vit":
            return "bias" in name or "cls" in name or "norm" in name or param.ndim == 1

        elif skip_type == "dino":
            return "bias" in name or param.ndim == 1

    return check_wd_skip_fn


def make_optimizer(train_conf, parameters):
    lr = train_conf.base_lr * train_conf.dataloader.batch_size / 256

    return train_conf.optimizer.make(parameters, lr=lr)


def make_scheduler(train_conf, optimizer, epoch_len):
    warmup = train_conf.scheduler.warmup * epoch_len
    n_iter = epoch_len * train_conf.epoch
    lr = train_conf.base_lr * train_conf.dataloader.batch_size / 256

    if train_conf.scheduler.type == "exp_epoch":
        return train_conf.scheduler.make(
            optimizer, epoch_len, lr=lr, max_iter=train_conf.epoch, warmup=warmup
        )

    else:
        return train_conf.scheduler.make(optimizer, lr=lr, n_iter=n_iter, warmup=warmup)


def repeated_sampler(sampler):
    epoch = 0

    while True:
        for i in sampler:
            yield i

        epoch += 1
        sampler.set_epoch(epoch)


class ExternalSource:
    def __init__(self, dataset, batch_size, shuffle, distributed):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = dist.data_sampler(dataset, shuffle=True, distributed=distributed)

    def __iter__(self):
        self.generator = repeated_sampler(self.sampler)

        return self

    def __next__(self):
        images, labels = [], []

        for _ in range(self.batch_size):
            img, label = self.dataset[next(self.generator)]
            images.append(np.frombuffer(img, dtype=np.uint8))
            labels.append(label)

        return images, torch.tensor(labels, dtype=torch.int64)


# @pipeline_def
def dali_pipeline(source, image_size, training, cpu=False):
    images, labels = fn.external_source(source=source, num_outputs=2)

    if cpu:
        device = "cpu"
        images = fn.decoders.image(images, device=device)

    else:
        device = "gpu"
        images = fn.decoders.image(
            images,
            device="mixed",
            device_memory_padding=211025920,
            host_memory_padding=140544512,
        )

    if training:
        images = fn.random_resized_crop(
            images,
            device=device,
            size=image_size,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
        )
        coin = fn.random.coin_flip(0.5)
        images = fn.flip(images, horizontal=coin)

    else:
        pass

    return images, labels


class DALIWrapper:
    def __init__(self, pipeline):
        self.dataloader = DALIClassificationIterator(pipeline)

    def __iter__(self):
        self.iterator = iter(self.dataloader)

        return self

    def __next__(self):
        data = next(self.iterator)
        image = data[0]["data"]
        label = data[0]["label"]


def make_dali_dataloader(
    path, train_size, valid_size, train_set, valid_set, batch, distributed, n_worker
):
    pass


def make_augment_dataset(path, train_transform, valid_transform):
    train_dir = os.path.join(nsml.DATASET_PATH, path, "train.lmdb")
    valid_dir = os.path.join(nsml.DATASET_PATH, path, "valid.lmdb")

    train_set = LMDBDataset(train_dir, train_transform)
    valid_set = LMDBDataset(valid_dir, valid_transform)

    return train_set, valid_set


def make_dataset(
    path, train_size, valid_size, randaug_params, mix_params, erasing, verbose=True
):
    train_dir = os.path.join(nsml.DATASET_PATH, path, "train.lmdb")
    valid_dir = os.path.join(nsml.DATASET_PATH, path, "valid.lmdb")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_list = [
        transforms.RandomResizedCrop(train_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandAugment(**randaug_params),
        transforms.ToTensor(),
        normalize,
    ]

    if erasing > 0:
        transform_list += [
            RandomErasing(
                erasing, mode="pixel", max_count=1, num_splits=0, device="cpu"
            )
        ]

    if mix_params["mix_before_aug"]:
        preprocess = transform_list[:2]
        postprocess = transform_list[2:]

    else:
        preprocess = transform_list
        postprocess = []

    if verbose:
        logger = get_logger()

        log = f"""Transforms
Transform before Mixes:
{preprocess}
Mixes: mixup={mix_params["mixup"]}, cutmix={mix_params["cutmix"]}"""

        if mix_params["mix_before_aug"]:
            log += f"""
Transform after Mixes:
{postprocess}"""

        logger.info(log)

    train_preprocess = transforms.Compose(preprocess)
    train_postprocess = transforms.Compose(postprocess)

    train_set = LMDBDataset(train_dir, train_preprocess)
    train_set = MixDataset(
        train_set, train_postprocess, mix_params["mixup"], mix_params["cutmix"]
    )

    valid_preprocess = transforms.Compose(
        [
            transforms.Resize(valid_size + 32, interpolation=Image.BICUBIC),
            transforms.CenterCrop(valid_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    valid_set = LMDBDataset(valid_dir, valid_preprocess)

    return train_set, valid_set


def make_dataset_cuda(path, train_size, valid_size, randaug_params, mixup, cutmix):
    train_dir = os.path.join(nsml.DATASET_PATH, path, "train.lmdb")
    valid_dir = os.path.join(nsml.DATASET_PATH, path, "valid.lmdb")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_preprocess = transforms.Compose(
        [
            transforms.RandomResizedCrop(train_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
    )
    train_postprocess = transforms.Compose(
        [RandAugment(**randaug_params), transforms.ToTensor(), normalize]
    )

    train_set = LMDBDataset(train_dir, train_preprocess)
    train_set = MixDataset(train_set, train_postprocess, mixup, cutmix)

    valid_preprocess = transforms.Compose(
        [
            transforms.Resize(valid_size + 32, interpolation=Image.BICUBIC),
            transforms.CenterCrop(valid_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    valid_set = LMDBDataset(valid_dir, valid_preprocess)

    return train_set, valid_set


def make_dataloader(train_set, valid_set, batch, distributed, n_worker):
    batch_size = batch // dist.get_world_size()

    train_sampler = dist.data_sampler(train_set, shuffle=True, distributed=distributed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=n_worker
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        sampler=dist.data_sampler(valid_set, shuffle=False, distributed=distributed),
        num_workers=n_worker,
    )

    return train_loader, valid_loader, train_sampler


def lerp(start, end, stage, max_stage):
    return start + (end - start) * (stage / (max_stage - 1))


def progressive_adaptive_regularization(
    stage,
    max_stage,
    train_sizes,
    valid_sizes,
    randaug_layers,
    randaug_magnitudes,
    mixups,
    cutmixes,
    dropouts,
    drop_paths,
    verbose=True,
):
    train_size = int(lerp(*train_sizes, stage, max_stage))
    valid_size = int(lerp(*valid_sizes, stage, max_stage))
    randaug_layer = int(lerp(*randaug_layers, stage, max_stage))
    randaug_magnitude = lerp(*randaug_magnitudes, stage, max_stage)
    mixup = lerp(*mixups, stage, max_stage)
    cutmix = lerp(*cutmixes, stage, max_stage)
    dropout = lerp(*dropouts, stage, max_stage)
    drop_path = lerp(*drop_paths, stage, max_stage)

    if verbose:
        logger = get_logger()
        log = f"""Progressive Training with Adaptive Regularization
Stage: {stage + 1} / {max_stage}
Image Size: train={train_size}, valid={valid_size}
RandAugment: n_augment={randaug_layer}, magnitude={randaug_magnitude}
Mixup: {mixup}, Cutmix: {cutmix}, Dropout={dropout}, DropPath={drop_path}"""
        logger.info(log)

    return SimpleNamespace(
        train_size=train_size,
        valid_size=valid_size,
        randaug_layer=randaug_layer,
        randaug_magnitude=randaug_magnitude,
        mixup=mixup,
        cutmix=cutmix,
        dropout=dropout,
        drop_path=drop_path,
    )
