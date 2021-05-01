import torch
from torch import nn
from torch.nn import functional as F

from models import layer


class NFBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        activation,
        ratio=0.5,
        se_ratio=0.5,
        group_size=1,
        stride=1,
        alpha=0.2,
        beta=1,
        stochastic_depth=None,
    ):
        super().__init__()

        in_ch = in_channel
        out_ch = out_channel
        ch = int(out_ch * ratio)
        groups = ch // group_size
        ch = group_size * groups

        self.alpha = alpha
        self.beta = beta

        self.conv1 = layer.WSConv2d(in_ch, ch, 1)
        self.conv2 = layer.WSConv2d(ch, ch, 3, padding=1, stride=stride, groups=groups)
        self.conv3 = layer.WSConv2d(ch, ch, 3, padding=1, groups=groups)
        self.conv4 = layer.WSConv2d(ch, out_ch, 1)

        self.skip = None
        if stride > 1 or in_ch != out_ch:
            skip_layers = []
            if stride > 1:
                skip_layers.append(nn.AvgPool2d(stride))

            skip_layers.append(layer.WSConv2d(in_ch, out_ch, 1))
            self.skip = nn.Sequential(*skip_layers)

        self.se = layer.SqueezeExcite(out_ch, out_ch, se_ratio)

        self.stochastic_depth = None
        if stochastic_depth is not None:
            self.stochastic_depth = layer.StochasticDepth(stochastic_depth)

        self.activation = activation
        self.skip_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        out = self.beta * self.activation(input)
        skip = input
        if self.skip is not None:
            skip = self.skip(out)

        out = self.conv1(out)
        out = self.conv2(self.activation(out))
        out = self.conv3(self.activation(out))
        out = self.conv4(self.activation(out))
        out = (2 * self.se(out)) * out

        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out)

        out = self.skip_gain * out

        return self.alpha * out + skip


class NFNet(nn.Module):
    def __init__(
        self,
        n_class,
        channels,
        depths,
        groups,
        width=1,
        alpha=0.2,
        stochastic_depth=0.1,
        dropout=0,
        activation="silu",
    ):
        super().__init__()

        self.activation = layer.ScaledActivation(activation)

        ch = channels[0] // 2
        self.stem = nn.Sequential(
            layer.WSConv2d(3, 16, 3, stride=2, padding=1),
            self.activation,
            layer.WSConv2d(16, 32, 3, padding=1),
            self.activation,
            layer.WSConv2d(32, 64, 3, padding=1),
            self.activation,
            layer.WSConv2d(64, ch, 3, stride=2, padding=1),
        )

        expected_std = 1
        strides = [1, 2, 2, 2]
        blocks = []
        n_blocks = sum(depths)
        index = 0

        for (channel, depth, group, stride) in zip(channels, depths, groups, strides):
            for i in range(depth):
                beta = 1 / expected_std
                block_stochastic_depth = stochastic_depth * index / n_blocks
                out_ch = int(channel * width)
                blocks.append(
                    NFBlock(
                        ch,
                        out_ch,
                        self.activation,
                        0.5,
                        0.5,
                        group,
                        stride=stride if i == 0 else 1,
                        beta=beta,
                        alpha=alpha,
                        stochastic_depth=block_stochastic_depth,
                    )
                )
                ch = out_ch
                index += 1

                if i == 0:
                    expected_std = 1

                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.blocks = nn.ModuleList(blocks)

        self.final_conv = layer.WSConv2d(ch, ch * 2, 1)

        linear = nn.Linear(ch * 2, n_class)
        nn.init.normal_(linear.weight, std=0.01)
        nn.init.zeros_(linear.bias)
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Dropout(dropout), linear
        )

    def forward(self, input):
        out = self.stem(input)

        for block in self.blocks:
            out = block(out)

        out = self.activation(self.final_conv(out))
        out = self.linear(out)

        return out
