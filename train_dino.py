import argparse
import os
import sys
import math
from time import perf_counter

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.nn.modules import adaptive
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tensorfn import load_arg_config, distributed as dist, Checker, get_logger

import models
from config import ImageNetConfig
from train_util import (
    accumulate,
    add_weight_decay,
    accuracy,
    Meter,
    cosine_schedule,
    cancel_last_layer_grad,
)
from factory import (
    make_augment_dataset,
    make_dataloader,
    make_optimizer,
    make_scheduler,
    progressive_adaptive_regularization,
    wd_skip_fn,
)
from transforms import DINOAugment
from optimizer import adaptive_grad_clip
from loss import DINOLoss


def main(conf):
    device = "cuda"
    conf.distributed = conf.n_gpu > 1
    torch.backends.cudnn.benchmark = True

    logger = get_logger(mode=conf.logger)
    logger.info(conf.dict())

    student = conf.arch.make().to(device)
    student.set_drop_path(conf.task.student_drop_path)
    teacher = conf.arch.make().to(device)

    logger.info(student)

    if conf.distributed:
        teacher = nn.parallel.DistributedDataParallel(
            teacher,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
        student = nn.parallel.DistributedDataParallel(
            student,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
        teacher_module = teacher.module
        student_module = student.module

        teacher_module.load_state_dict(student_module.state_dict())

    else:
        teacher_module = teacher
        student_module = student

        teacher_module.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    grad_accum = conf.training.grad_accumulation

    train_set, valid_set = make_augment_dataset(
        conf.dataset_path,
        DINOAugment(
            conf.task.global_crop_size,
            conf.task.local_crop_size,
            conf.task.global_crop_scale,
            conf.task.local_crop_scale,
            conf.task.n_local_crop,
        ),
        None,
    )

    batch_size = conf.training.dataloader.batch_size // grad_accum

    train_loader, valid_loader, train_sampler = make_dataloader(
        train_set,
        valid_set,
        batch_size,
        conf.distributed,
        conf.training.dataloader.num_workers,
    )

    criterion_train = DINOLoss(
        conf.arch.dim_head_out,
        conf.task.n_local_crop + 2,
        conf.task.warmup_teacher_temperature,
        conf.task.teacher_temperature,
        conf.task.warmup_teacher_temperature_epoch,
        conf.training.epoch,
    ).to(device)

    parameters, names = add_weight_decay(
        student.named_parameters(),
        conf.training.weight_decay,
        wd_skip_fn(conf.training.wd_skip),
    )

    def make_scheduler(train_conf, optimizer, epoch_len):
        warmup = train_conf.scheduler.warmup * epoch_len
        n_iter = epoch_len * train_conf.epoch
        lr = train_conf.base_lr * train_conf.dataloader.batch_size / 256

        if train_conf.scheduler.type == "exp_epoch":
            return train_conf.scheduler.make(
                optimizer, epoch_len, lr=lr, max_iter=train_conf.epoch, warmup=warmup
            )

        else:
            return train_conf.scheduler.make(
                optimizer, lr=lr, n_iter=n_iter, warmup=warmup
            )

    optimizer = make_optimizer(conf.training, parameters)
    epoch_len = math.ceil(len(train_loader) / grad_accum)
    scheduler = make_scheduler(conf.training, optimizer, epoch_len)
    wd_schedule = cosine_schedule(
        conf.training.weight_decay,
        conf.task.weight_decay_end,
        epoch_len * conf.training.epoch,
    )
    momentum_schedule = cosine_schedule(
        conf.task.teacher_momentum, 1, epoch_len * conf.training.epoch
    )

    scaler = amp.GradScaler(enabled=conf.fp16)

    checker = conf.checker.make()

    step = 0

    for epoch in range(conf.training.epoch):
        if conf.distributed:
            train_sampler.set_epoch(epoch)

        train(
            conf,
            step,
            epoch,
            train_loader,
            teacher,
            student,
            criterion_train,
            optimizer,
            scheduler,
            wd_schedule,
            momentum_schedule,
            scaler,
            grad_accum,
            checker,
        )
        step += epoch_len

        try:
            checker.checkpoint(
                {
                    "student": student_module.state_dict(),
                    "teacher": teacher_module.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf.dict(),
                },
                f"epoch-{str(epoch + 1).zfill(3)}.pt",
            )

        except Exception as e:
            print(e)


def train(
    conf,
    step,
    epoch,
    loader,
    teacher,
    student,
    criterion,
    optimizer,
    scheduler,
    wd_schedule,
    momentum_schedule,
    scaler,
    grad_accum,
    checker,
):
    device = "cuda"

    batch_time = Meter()
    data_time = Meter()
    losses = Meter()

    student.train()

    agc_params = [p[1] for p in student.named_parameters() if "linear" not in p[0]]
    params = list(student.parameters())

    logger = get_logger(mode=conf.logger)

    start = perf_counter()
    for i, (inputs, _) in enumerate(loader):
        # measure data loading time
        inputs = [i.to(device) for i in inputs]
        data_time.update(perf_counter() - start)

        with amp.autocast(enabled=conf.fp16):
            with torch.no_grad():
                teacher_out = teacher(inputs[:2])

            student_out = student(inputs)

            loss = criterion(student_out, teacher_out, epoch) / grad_accum

        losses.update(loss.item() * grad_accum, inputs[0].shape[0])

        scaler.scale(loss).backward()

        for param_group in optimizer.param_groups:
            if "no_decay" not in param_group:
                param_group["weight_decay"] = wd_schedule[step]

        if ((i + 1) % grad_accum == 0) or (i + 1) == len(loader):
            if conf.training.agc > 0 or conf.training.clip_grad_norm > 0:
                if conf.fp16:
                    scaler.unscale_(optimizer)

                if conf.training.agc > 0:
                    adaptive_grad_clip(agc_params, conf.training.agc)

                if conf.training.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(params, conf.training.clip_grad_norm)

            cancel_last_layer_grad(epoch, student, conf.task.freeze_last_layer)

            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                m = momentum_schedule[step]

                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.detach().mul_(m).add_(param_q.detach(), alpha=1 - m)

        batch_time.update(perf_counter() - start)
        start = perf_counter()

        if dist.is_primary() and i % conf.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]

            """logger.info(
                f"epoch: {epoch} ({i}/{len(loader)}); time: {batch_time.val:.3f} ({batch_time.avg:.2f}); "
                f"data: {data_time.val:.3f} ({data_time.avg:.2f}); "
                f"loss: {losses.val:.3f} ({losses.avg:.3f}); "
                f"lr: {lr:.5f}; "
                f"wd: {wd_schedule[step]:4f}; "
                f"moment: {momentum_schedule[step]:.4f}"
            )"""

            checker.log(
                step=step,
                weight_decay=wd_schedule[step],
                momentum=momentum_schedule[step],
                loss=losses.avg,
                lr=optimizer.param_groups[0]["lr"],
            )

        step += 1

    return losses


if __name__ == "__main__":
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    conf = load_arg_config(ImageNetConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )

