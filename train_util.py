import math

import torch


def cosine_schedule(base, final, step, warmup=0, warmup_start=0):
    if warmup > 0:
        warmup_schedule = torch.linspace(warmup_start, base, warmup)

    else:
        warmup_schedule = torch.tensor([])

    iters = torch.arange(step - warmup)
    schedule = torch.tensor(
        [
            final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )
    schedule = torch.cat((warmup_schedule, schedule)).tolist()

    return schedule


def cancel_last_layer_grad(epoch, model, freeze):
    if epoch >= freeze:
        return

    for n, p in model.named_parameters():
        if "last" in n:
            p.grad = None


class Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def accumulate(model1, model2, decay=0.99999, ema_bn=False):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    if ema_bn:
        buf1 = dict(model1.named_buffers())
        buf2 = dict(model2.named_buffers())

        for k in buf1.keys():
            if "running_mean" in k or "running_var" in k:
                buf1[k].data.mul_(decay).add_(buf2[k].data, alpha=1 - decay)


def add_weight_decay(named_parameters, weight_decay, check_skip_fn):
    decay = []
    decay_names = []
    no_decay = []
    no_decay_names = []

    for n, p in named_parameters:
        if not p.requires_grad:
            continue

        if check_skip_fn(n, p):
            no_decay.append(p)
            no_decay_names.append(n)

        else:
            decay.append(p)
            decay_names.append(n)

    return (
        (
            {"params": no_decay, "weight_decay": 0.0, "no_decay": True},
            {"params": decay, "weight_decay": weight_decay},
        ),
        (no_decay_names, decay_names),
    )

