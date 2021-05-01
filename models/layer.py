from collections import abc
from itertools import repeat

import torch
from torch import nn
from torch.nn import functional as F


def ensure_tuple(x, n_item):
    if isinstance(x, abc.Iterable):
        try:
            if len(x) != n_item:
                raise ValueError(
                    f"length of {x} (length: {len(x)}) does not match with the condition. expected length: {n_item}"
                )

        except TypeError:
            pass

        return x

    return tuple(repeat(x, n_item))


tuple2 = lambda x: ensure_tuple(x, 2)


activations = {
    "identity": lambda x: x,
    "gelu": lambda x: F.gelu(x) * 1.7015043497085571,
    "relu": lambda x: F.relu(x) * 1.7139588594436646,
    "silu": lambda x: F.silu(x) * 1.7881293296813965,
}


class ScaledActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.name = activation
        self.activation = activations[activation]

    def forward(self, input):
        return self.activation(input)

    def __repr__(self):
        return f"ScaledActivation({self.name})"


class WSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.gain = nn.Parameter(torch.ones(out_channels))
        self.eps = eps ** 2
        fan_in = torch.numel(self.weight[0])
        self.scale = fan_in ** -0.5
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def forward(self, input):
        weight = F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        gain = self.gain.view([-1] + [1] * (weight.ndim - 1))
        weight = gain * self.scale * weight

        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class StochasticDepth(nn.Module):
    def __init__(self, p, scale_by_keep=False):
        super().__init__()

        self.p = p
        self.scale_by_keep = scale_by_keep

    def forward(self, input):
        if not self.training:
            return input

        mask = input.new_empty([input.shape[0]] + [1] * (input.ndim - 1)).bernoulli_(
            1 - self.p
        )

        if self.scale_by_keep:
            input = input / (1 - self.p)

        return input * mask

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale_by_keep={self.scale_by_keep})"
        )


class SqueezeExcite(nn.Sequential):
    def __init__(
        self, in_channel, out_channel, ratio=0.5, channel=None, activation=nn.ReLU
    ):
        if channel is None:
            channel = max(1, int(in_channel * ratio))

        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, channel, 1),
            activation(),
            nn.Conv2d(channel, out_channel, 1),
            nn.Sigmoid(),
        )


class GlobalContext(nn.Module):
    def __init__(
        self, in_channel, out_channel, ratio=0.25, channel=None, activation=nn.ReLU
    ):
        if channel is None:
            channel = max(1, int(in_channel * ratio))

        self.key = nn.Conv2d(in_channel, 1, 1)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1),
            nn.LayerNorm((channel, 1, 1)),
            activation(),
            nn.Conv2d(channel, out_channel, 1),
        )

    def forward(self, input):
        batch, channel, height, width = input.shape

        logit = self.key(input)
        attn = torch.softmax(logit.view(batch, 1, -1, 1), 2)  # N 1 HW 1
        value = input.view(batch, 1, channel, -1)  # N 1 C HW
        attn_pool = (value @ attn).view(batch, channel, 1, 1)

        context = self.layers(attn_pool)

        return context


class DropPath(nn.Module):
    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, input):
        if not self.training or self.p == 0:
            return input

        p = 1 - self.p
        mask = input.new_empty([input.shape[0]] + [1] * (input.ndim - 1)).bernoulli_(p)
        out = input / p * mask

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, in_dim, dim=None, out_dim=None, activation=nn.SiLU, dropout=0):
        dim = in_dim if dim is None else dim
        out_dim = in_dim if out_dim is None else out_dim

        super().__init__(
            nn.Linear(in_dim, dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )
