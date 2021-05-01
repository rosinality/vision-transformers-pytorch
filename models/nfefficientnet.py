import math

import torch
from torch import nn
from torch.nn import functional as F
from tensorfn.config import config_model
from pydantic import StrictFloat

from models.layer import StochasticDepth, WSConv2d, ScaledActivation


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
        alpha=0.2,
        beta=1,
    ):
        super().__init__()

        channel = in_channel * expand_ratio

        layers = []

        if fused:
            layers.append(
                WSConv2d(
                    in_channel,
                    channel,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )

        else:
            if expand_ratio != 1:
                layers += [WSConv2d(in_channel, channel, 1), ScaledActivation("silu")]

            layers.append(
                WSConv2d(
                    channel,
                    channel,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=channel,
                )
            )

        layers += [ScaledActivation("silu")]

        self.conv = nn.Sequential(*layers)

        if se is not None:
            """se_channel = max(1, int(in_channel * se))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, se_channel, 1),
                ScaledActivation("silu"),
                nn.Conv2d(se_channel, channel, 1),
                nn.Sigmoid(),
            )"""
            self.se = None

        else:
            self.se = None

        self.project = WSConv2d(channel, out_channel, 1)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.skip = skip and self.stride == 1 and self.in_channel == self.out_channel

        if self.skip:
            self.drop_path = StochasticDepth(drop_path)
            self.skip_gain = nn.Parameter(torch.tensor(0.0))

        self.alpha = alpha
        self.beta = beta

    def set_drop_path(self, p):
        if self.drop_path is not None:
            self.drop_path.p = p

    def forward(self, input):
        out = self.beta * input
        out = self.conv(out)

        if self.se is not None:
            out = 2 * self.se(out) * out

        project = self.project(out)
        if self.skip:
            out = self.drop_path(project)
            out = self.skip_gain * out
            out = self.alpha * out
            out = out + input
        else:
            # print("stride")
            out = project

        """acms = out.mean((0, 2, 3)).pow(2).mean()
        acv = out.var((0, 2, 3)).mean()
        racv = project.var((0, 2, 3)).mean()

        print(
            round(self.beta, 4),
            round(acms.item(), 4),
            round(acv.item(), 4),
            round(racv.item(), 4),
            sep="\t",
        )"""

        return out


class NFEfficientNet(nn.Module):
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
        alpha=0.2,
    ):
        super().__init__()

        stem_channel = round_filters(stem_channel, width, divisor, min_depth)

        activation = ScaledActivation("silu")

        self.stem = nn.Sequential(
            WSConv2d(3, stem_channel, 3, stride=2, padding=1), activation
        )

        blocks = []
        expected_std = 1
        second = False
        for config in block_configs:
            beta = 1 / expected_std
            # print(expected_std)

            config = config.copy()
            config["in_channel"] = round_filters(
                config["in_channel"], width, divisor, min_depth
            )
            config["out_channel"] = round_filters(
                config["out_channel"], width, divisor, min_depth
            )
            n_repeat = round_repeats(config.pop("n_repeat"), depth)
            config["alpha"] = alpha
            config["beta"] = beta

            blocks.append(MBConvBlock(**config))

            if not blocks[-1].skip:
                # print("skip")
                expected_std = 1.0
                second = True

            if n_repeat > 1:
                config["in_channel"] = config["out_channel"]
                config["stride"] = 1

            for i in range(n_repeat - 1):
                if not second:
                    expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5
                # print(i, expected_std)
                second = False
                beta = 1 / expected_std
                config["alpha"] = alpha
                config["beta"] = beta
                blocks.append(MBConvBlock(**config))

            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.blocks = nn.Sequential(*blocks)

        out_channel = config["out_channel"]

        head_channel = round_filters(head_channel, width, divisor, min_depth)
        self.head = nn.Sequential(WSConv2d(out_channel, head_channel, 1), activation)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.logit = nn.Linear(head_channel, n_class)
        nn.init.normal_(self.logit.weight, std=0.01)
        nn.init.zeros_(self.logit.bias)

    def set_dropout(self, dropout, drop_path):
        n_blocks = len(self.blocks)
        dp_rate = [drop_path * float(i) / n_blocks for i in range(n_blocks)]

        for block, dp in zip(self.blocks, dp_rate):
            block.set_drop_path(dp)

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


@config_model(namespace="model", use_type=True)
def nfefficientnet(width: StrictFloat, depth: StrictFloat):
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


@config_model(namespace="model", use_type=True)
def nfefficientnetv2(width: StrictFloat, depth: StrictFloat):
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

    return NFEfficientNet(
        1000,
        block_configs,
        width=width,
        divisor=divisor,
        min_depth=min_depth,
        depth=depth,
        stem_channel=24,
        head_channel=1792,
    )


def nfefficientnetv2_s():
    return nfefficientnetv2(1, 1)
