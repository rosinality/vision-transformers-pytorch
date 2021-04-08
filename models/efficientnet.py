import math

import torch
from torch import nn
from torch.nn import functional as F
from tensorfn.config import config_model
from pydantic import StrictFloat

from models.layer import DropPath


def round_filters(filters, width=None, divisor=None, min_depth=None):
    if width is None:
        return filters

    filters *= width
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, depth=None):
    if depth is None:
        return repeats

    return int(math.ceil(depth * repeats))


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        expand_ratio=1,
        se=0.25,
        skip=True,
        fused=False,
        drop_path=0,
    ):
        super().__init__()

        channel = in_channel * expand_ratio

        layers = []

        if fused:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    channel,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )

        else:
            if expand_ratio != 1:
                layers += [
                    nn.Conv2d(in_channel, channel, 1, bias=False),
                    nn.BatchNorm2d(channel, momentum=0.99),
                    nn.SiLU(),
                ]

            layers.append(
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=channel,
                    bias=False,
                )
            )

        layers += [nn.BatchNorm2d(channel, momentum=0.99), nn.SiLU()]

        self.conv = nn.Sequential(*layers)

        if se is not None:
            se_channel = max(1, int(in_channel * se))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, se_channel, 1),
                nn.SiLU(),
                nn.Conv2d(se_channel, channel, 1),
                nn.Sigmoid(),
            )

        else:
            self.se = None

        self.project = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=0.99),
        )

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.skip = skip and self.stride == 1 and self.in_channel == self.out_channel

        if self.skip:
            self.drop_path = DropPath(drop_path)

        else:
            self.drop_path = None

    def set_drop_path(self, p):
        if self.drop_path is not None:
            self.drop_path.p = p

    def forward(self, input):
        out = self.conv(input)

        if self.se is not None:
            out = self.se(out) * out

        out = self.project(out)

        if self.skip:
            out = self.drop_path(out) + input

        return out


class EfficientNet(nn.Module):
    def __init__(
        self,
        n_class,
        block_configs,
        width=None,
        divisor=None,
        min_depth=None,
        depth=None,
        stem_channel=32,
        head_channel=1280,
        dropout=0,
    ):
        super().__init__()

        stem_channel = round_filters(stem_channel, width, divisor, min_depth)

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channel, momentum=0.99),
            nn.SiLU(),
        )

        blocks = []
        for config in block_configs:
            config = config.copy()
            config["in_channel"] = round_filters(
                config["in_channel"], width, divisor, min_depth
            )
            config["out_channel"] = round_filters(
                config["out_channel"], width, divisor, min_depth
            )
            n_repeat = round_repeats(config.pop("n_repeat"), depth)

            blocks.append(MBConvBlock(**config))

            if n_repeat > 1:
                config["in_channel"] = config["out_channel"]
                config["stride"] = 1

            for i in range(n_repeat - 1):
                blocks.append(MBConvBlock(**config))

        self.blocks = nn.Sequential(*blocks)

        out_channel = config["out_channel"]

        head_channel = round_filters(head_channel, width, divisor, min_depth)
        self.head = nn.Sequential(
            nn.Conv2d(out_channel, head_channel, 1, bias=False),
            nn.BatchNorm2d(head_channel, momentum=0.99),
            nn.SiLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.logit = nn.Linear(head_channel, n_class)

    def set_dropout(self, dropout, drop_path):
        for block in self.blocks:
            block.set_drop_path(drop_path)

        self.dropout.p = dropout

    def forward(self, input):
        out = self.stem(input)
        out = self.blocks(out)
        out = self.head(out)
        out = self.avg_pool(out)
        out = out.squeeze(3).squeeze(2)
        out = self.dropout(out)
        out = self.logit(out)

        return out


@config_model(use_type=True)
def efficientnet(width: StrictFloat, depth: StrictFloat):
    divisor = 8
    min_depth = None

    block_configs = [
        {
            "n_repeat": 1,
            "kernel_size": 3,
            "stride": 1,
            "expand_ratio": 1,
            "in_channel": 32,
            "out_channel": 16,
        },
        {
            "n_repeat": 2,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 6,
            "in_channel": 16,
            "out_channel": 24,
        },
        {
            "n_repeat": 2,
            "kernel_size": 5,
            "stride": 2,
            "expand_ratio": 6,
            "in_channel": 24,
            "out_channel": 40,
        },
        {
            "n_repeat": 3,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 6,
            "in_channel": 40,
            "out_channel": 80,
        },
        {
            "n_repeat": 3,
            "kernel_size": 5,
            "stride": 1,
            "expand_ratio": 6,
            "in_channel": 80,
            "out_channel": 112,
        },
        {
            "n_repeat": 4,
            "kernel_size": 5,
            "stride": 2,
            "expand_ratio": 6,
            "in_channel": 112,
            "out_channel": 192,
        },
        {
            "n_repeat": 1,
            "kernel_size": 3,
            "stride": 1,
            "expand_ratio": 6,
            "in_channel": 192,
            "out_channel": 320,
        },
    ]

    return EfficientNet(
        1000,
        block_configs,
        width=width,
        divisor=divisor,
        min_depth=min_depth,
        depth=depth,
    )


def efficientnet_b1():
    return efficientnet(1.0, 1.1)


def efficientnet_b2():
    return efficientnet(1.1, 1.2)


def efficientnet_b3():
    return efficientnet(1.2, 1.4)


def efficientnet_b4():
    return efficientnet(1.4, 1.8)


def efficientnet_b5():
    return efficientnet(1.6, 2.2)


def efficientnet_b6():
    return efficientnet(1.8, 2.6)


def efficientnet_b7():
    return efficientnet(2.0, 3.1)


@config_model(use_type=True)
def efficientnetv2(width: StrictFloat, depth: StrictFloat):
    divisor = 8
    min_depth = None

    block_configs = [
        {
            "n_repeat": 2,
            "kernel_size": 3,
            "stride": 1,
            "expand_ratio": 1,
            "in_channel": 24,
            "out_channel": 24,
            "fused": True,
            "se": None,
        },
        {
            "n_repeat": 4,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 4,
            "in_channel": 24,
            "out_channel": 48,
            "fused": True,
            "se": None,
        },
        {
            "n_repeat": 4,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 4,
            "in_channel": 48,
            "out_channel": 64,
            "fused": True,
            "se": None,
        },
        {
            "n_repeat": 6,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 4,
            "in_channel": 64,
            "out_channel": 128,
        },
        {
            "n_repeat": 9,
            "kernel_size": 3,
            "stride": 1,
            "expand_ratio": 6,
            "in_channel": 128,
            "out_channel": 160,
        },
        {
            "n_repeat": 15,
            "kernel_size": 3,
            "stride": 2,
            "expand_ratio": 6,
            "in_channel": 160,
            "out_channel": 272,
        },
    ]

    return EfficientNet(
        1000,
        block_configs,
        width=width,
        divisor=divisor,
        min_depth=min_depth,
        depth=depth,
        stem_channel=24,
        head_channel=1792,
    )


def efficientnetv2_s():
    return efficientnetv2(1, 1)
