import argparse
import os
import sys
from time import perf_counter

import torch
from torch import nn, optim
from torch.nn.modules import adaptive
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tensorfn import load_arg_config, distributed as dist, Checker

import models
from config import ImageNetConfig
from train_util import accumulate, add_weight_decay, accuracy, Meter
from factory import (
    make_dataset,
    make_dataloader,
    make_optimizer,
    make_scheduler,
    progressive_adaptive_regularization,
    wd_skip_fn,
)
from loss import MixLoss


def make_progressive_loader(stage, model, conf):
    adapt = progressive_adaptive_regularization(
        stage,
        conf.training.epoch // conf.training.progressive.step,
        conf.training.progressive.train_sizes,
        conf.training.progressive.valid_sizes,
        conf.training.progressive.randaug_layers,
        conf.training.progressive.randaug_magnitudes,
        conf.training.progressive.mixups,
        conf.training.progressive.cutmixes,
        conf.training.progressive.dropouts,
        conf.training.progressive.drop_paths,
        conf.training.progressive.verbose,
    )
    train_set, valid_set = make_dataset(
        conf.dataset_path,
        adapt.train_size,
        adapt.valid_size,
        {
            "n_augment": adapt.randaug_layer,
            "magnitude": adapt.randaug_magnitude,
            "increasing": conf.training.randaug_increasing,
            "magnitude_std": conf.training.randaug_magnitude_std,
        },
        adapt.mixup,
        adapt.cutmix,
    )
    train_loader, valid_loader = make_dataloader(
        train_set,
        valid_set,
        conf.training.dataloader.batch_size,
        conf.distributed,
        conf.training.dataloader.num_workers,
    )

    try:
        model.set_dropout(adapt.dropout, adapt.drop_path)

    except:
        pass

    return train_loader, valid_loader


def main(conf):
    device = "cuda"
    conf.distributed = conf.n_gpu > 1
    torch.backends.cudnn.benchmark = True

    model = conf.arch.make().to(device)
    model_ema = conf.arch.make().to(device)

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
        model_module = model.module

        accumulate(model_ema, model_module, 0)

    else:
        model_module = model
        accumulate(model_ema, model, 0)

    if conf.training.progressive.step > 0:
        progressive_stage = 0
        train_loader, valid_loader = make_progressive_loader(
            progressive_stage, model_module, conf
        )

    else:
        train_set, valid_set = make_dataset(
            conf.dataset_path,
            conf.training.train_size,
            conf.training.valid_size,
            {
                "n_augment": conf.training.randaug_layer,
                "magnitude": conf.training.magnitude,
                "increasing": conf.training.randaug_increasing,
                "magnitude_std": conf.training.randaug_magnitude_std,
            },
            conf.training.mixup,
            conf.training.cutmix,
        )
        train_loader, valid_loader = make_dataloader(
            train_set,
            valid_set,
            conf.training.dataloader.batch_size,
            conf.distributed,
            conf.training.dataloader.num_workers,
        )

    criterion_train = MixLoss(eps=0.1)
    criterion_valid = nn.CrossEntropyLoss()

    parameters, names = add_weight_decay(
        model.named_parameters(),
        conf.training.weight_decay,
        wd_skip_fn(conf.training.wd_skip),
    )

    optimizer = make_optimizer(conf.training, parameters)
    epoch_len = len(train_loader)
    scheduler = make_scheduler(conf.training, optimizer, epoch_len)

    step = 0

    def checker_save(filename, *args):
        torch.save(
            {
                "model": model_module.state_dict(),
                "ema": model_ema.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "conf": conf,
            },
            filename,
        )

    checker = conf.checker.make(checker_save)
    checker.save("test")

    for epoch in range(conf.training.epoch):
        train(
            conf,
            step,
            epoch,
            train_loader,
            model,
            model_ema,
            criterion_train,
            optimizer,
            scheduler,
        )
        step += epoch_len

        if conf.training.ema == 0:
            prec1, prec5, losses = valid(
                conf, valid_loader, model_module, criterion_valid
            )

        else:
            prec1, prec5, losses = valid(conf, valid_loader, model_ema, criterion_valid)

        checker.log(
            step=epoch + 1,
            prec1=prec1,
            prec5=prec5,
            loss=losses.avg,
            lr=optimizer.param_groups[0]["lr"],
        )
        checker.save(f"epoch-{str(epoch + 1).zfill(3)}")

        if (
            conf.training.progressive.step > 0
            and (epoch + 1) % conf.training.progressive.step == 0
        ):
            progressive_stage += 1

            if (
                progressive_stage
                < conf.training.epoch // conf.training.progressive.step
            ):
                train_loader, valid_loader = make_progressive_loader(
                    progressive_stage, model_module, conf
                )


def train(conf, step, epoch, loader, model, model_ema, criterion, optimizer, scheduler):
    device = "cuda"

    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    model.train()

    agc_params = [p[1] for p in model.named_parameters() if "linear" not in p[0]]
    params = list(model.parameters())

    start = perf_counter()
    for i, (input, label1, label2, ratio) in enumerate(loader):
        # measure data loading time
        input = input.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        ratio = ratio.to(device=device, dtype=torch.float32)
        data_time.update(perf_counter() - start)

        out = model(input)
        loss = criterion(out, label1, label2, ratio)

        prec1, prec5 = accuracy(out, label1, topk=(1, 5))
        batch = input.shape[0]
        losses.update(loss.item(), batch)
        top1.update(prec1.item(), batch)
        top5.update(prec5.item(), batch)

        optimizer.zero_grad()
        loss.backward()

        if conf.training.agc > 0:
            adaptive_grad_clip(agc_params, args.agc)

        if conf.training.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(params, conf.training.clip_grad_norm)

        scheduler.step()
        optimizer.step()
        t = step + i

        if conf.training.ema > 0:
            if conf.distributed:
                model_module = model.module

            else:
                model_module = model

            accumulate(
                model_ema,
                model_module,
                min(conf.training.ema, (1 + t) / (10 + t)),
                ema_bn=conf.training.ema_bn,
            )

        batch_time.update(perf_counter() - start)
        start = perf_counter()

        if dist.is_primary() and i % conf.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch: {epoch} ({i}/{len(loader)}); time: {batch_time.val:.3f} ({batch_time.avg:.2f}); "
                f"data: {data_time.val:.3f} ({data_time.avg:.2f}); "
                f"loss: {losses.val:.3f} ({losses.avg:.3f}); "
                f"prec@1: {top1.val:.2f} ({top1.avg:.2f}); "
                f"prec@5: {top5.val:.2f} ({top5.avg:.2f})"
            )

    return losses


@torch.no_grad()
def valid(conf, loader, model, criterion):
    device = "cuda"

    batch_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    model.eval()

    start = perf_counter()
    for i, (input, label) in enumerate(loader):
        input = input.to(device)
        label = label.to(device)

        out = model(input)
        loss = criterion(out, label)
        prec1, prec5 = accuracy(out, label, topk=(1, 5))
        batch = input.shape[0]

        loss_dict = {
            "prec1": prec1 * batch,
            "prec5": prec5 * batch,
            "loss": loss * batch,
            "batch": torch.tensor(batch, dtype=torch.float32).to(device),
        }
        loss_reduced = dist.reduce_dict(loss_dict, average=False)
        batch = loss_reduced["batch"].to(torch.int64).item()
        losses.update(loss_reduced["loss"].item() / batch, batch)
        top1.update(loss_reduced["prec1"].item() / batch, batch)
        top5.update(loss_reduced["prec5"].item() / batch, batch)

        batch_time.update(perf_counter() - start)
        start = perf_counter()

        if dist.is_primary() and i % conf.log_freq == 0:
            print(
                f"valid: {i}/{len(loader)}; time: {batch_time.val:.3f} ({batch_time.avg:.3f}); "
                f"loss: {losses.val:.4f} ({losses.avg:.4f}); "
                f"prec@1: {top1.val:.3f} ({top1.avg:.3f}); "
                f"prec@5: {top5.val:.3f} ({top5.avg:.3f})"
            )

    if dist.is_primary():
        print(f"validation finished: prec@1 {top1.avg:.3f}, prec@5 {top5.avg:.3f}")

    return top1.avg, top5.avg, losses


if __name__ == "__main__":
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    conf = load_arg_config(ImageNetConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )

