import math
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tensorfn.config import config_model
from pydantic import StrictInt, StrictFloat

from .layer import DropPath, tuple2

LayerNorm = lambda x: nn.LayerNorm(x, eps=1e-6)


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, in_dim, dim=None, out_dim=None, activation=nn.SiLU, dropout=0):
        dim = in_dim if dim is None else dim
        out_dim = in_dim if out_dim is None else out_dim

        super().__init__(
            nn.Linear(in_dim, dim),
            activation(),
            nn.Linear(dim, out_dim),
            nn.Dropout(dropout),
        )


def patchify(input, size):
    batch, height, width, dim = input.shape

    return (
        input.view(batch, height // size, size, width // size, size, dim)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height // size, width // size, -1)
    )


class PositionalEncodingGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj = nn.Conv2d(dim, dim, 3, padding=1, bias=False, groups=dim)

    def forward(self, input):
        out = input.permute(0, 3, 1, 2)
        out = self.proj(out) + out
        out = out.permute(0, 2, 3, 1)

        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, n_head, reduction=1, dropout=0):
        super().__init__()

        self.dim_head = dim // n_head
        self.n_head = n_head

        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_kv = nn.Linear(dim, dim * 2, bias=False)

        self.linear = nn.Linear(dim, dim)
        self.dropout = dropout
        self.reduction = reduction

        if self.reduction > 1:
            self.reduce_conv = nn.Conv2d(
                dim, dim, self.reduction, stride=self.reduction
            )

    def forward(self, input):
        batch_size, height, width, _ = input.shape

        def reshape(input):
            return input.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(
                1, 2
            )

        query = reshape(self.linear_q(input))

        if self.reduction > 1:
            dim = input.shape[-1]
            reduc = input.transpose(1, 2).reshape(batch_size, dim, height, width)
            reduc = self.reduce_conv(reduc).reshape(batch_size, dim, -1).transpose(1, 2)
            kv = reduc

        else:
            kv = input

        key, value = self.linear_kv(kv).chunk(2, dim=2)
        key = reshape(key).transpose(2, 3)
        value = reshape(value)

        score = query @ key / math.sqrt(self.dim_head)

        attn = F.softmax(score, 3)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ value

        out = out.transpose(1, 2).reshape(
            batch_size, height, width, self.dim_head * self.n_head
        )
        out = self.linear(out)

        return out


class MultiHeadedLocalAttention(nn.Module):
    def __init__(self, dim, n_head, dim_head, window_size, dropout=0):
        super().__init__()

        self.dim_head = dim_head
        self.n_head = n_head

        self.weight = nn.Linear(dim, n_head * dim_head * 3, bias=True)
        self.linear = nn.Linear(n_head * dim_head, dim)

        self.window_size = window_size
        self.dropout = dropout

    def forward(self, input):
        batch, height, width, dim = input.shape
        h_stride = height // self.window_size
        w_stride = width // self.window_size
        window = self.window_size

        def reshape(input):
            return (
                input.reshape(
                    batch,
                    h_stride,
                    window,
                    w_stride,
                    window,
                    self.n_head,
                    self.dim_head,
                )
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(batch, -1, self.n_head, window * window, self.dim_head)
            )

        query, key, value = self.weight(input).chunk(3, dim=-1)  # B, S, H, W^2, D

        query = reshape(query)
        key = reshape(key).transpose(-2, -1)
        value = reshape(value)

        score = query @ key / math.sqrt(self.dim_head)  # B, S, H, W^2, W^2

        attn = F.softmax(score, -1)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ value  # B, S, H, W^2, D

        out = (
            out.view(
                batch, h_stride, w_stride, self.n_head, window, window, self.dim_head
            )
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(batch, height, width, self.n_head * self.dim_head)
        )
        out = self.linear(out)

        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        dim_head,
        dim_ff,
        window_size,
        activation=nn.SiLU,
        drop_ff=0,
        drop_attn=0,
        drop_path=0,
    ):
        super().__init__()

        self.norm_attn_local = LayerNorm(dim)
        self.attn_local = MultiHeadedLocalAttention(
            dim, n_head, dim_head, window_size, drop_attn
        )
        self.norm_ff_local = LayerNorm(dim)
        self.ff_local = PositionwiseFeedForward(
            dim, dim_ff, activation=activation, dropout=drop_ff
        )

        self.norm_attn_global = LayerNorm(dim)
        self.attn_global = MultiHeadedAttention(dim, n_head, window_size, drop_attn)
        self.norm_ff_global = LayerNorm(dim)
        self.ff_global = PositionwiseFeedForward(
            dim, dim_ff, activation=activation, dropout=drop_ff
        )

        self.drop_path = DropPath(drop_path)

    def set_drop_path(self, p):
        self.drop_path.p = p

    def forward(self, input):
        out = input + self.drop_path(self.attn_local(self.norm_attn_local(input)))
        out = out + self.drop_path(self.ff_local(self.norm_ff_local(out)))
        out = out + self.drop_path(self.attn_global(self.norm_attn_global(out)))
        out = out + self.drop_path(self.ff_global(self.norm_ff_global(out)))

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, window_size):
        super().__init__()

        self.window_size = window_size
        self.linear = nn.Linear(in_dim * window_size * window_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, input):
        out = patchify(input, self.window_size)
        out = self.linear(out)
        out = self.norm(out)

        return out


def reduce_size(size, reduction):
    return (size[0] // reduction, size[1] // reduction)


@config_model(name="twins_svt", namespace="model", use_type=True)
class TwinsSVT(nn.Module):
    def __init__(
        self,
        n_class: StrictInt,
        depths: Tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        dims: Tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        dim_head: StrictInt,
        n_heads: Tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        dim_ffs: Tuple[StrictInt, StrictInt, StrictInt, StrictInt],
        window_size: StrictInt,
        drop_ff: StrictFloat = 0.0,
        drop_attn: StrictFloat = 0.0,
        drop_path: StrictFloat = 0.0,
    ):
        super().__init__()

        self.depths = depths

        def make_block(i, in_dim, reduction):
            return self.make_block(
                depths[i],
                in_dim,
                dims[i],
                n_heads[i],
                dim_head,
                dim_ffs[i],
                window_size,
                reduction,
                drop_ff,
                drop_attn,
            )

        self.block1 = make_block(0, 3, 4)
        self.block2 = make_block(1, dims[0], 2)
        self.block3 = make_block(2, dims[1], 2)
        self.block4 = make_block(3, dims[2], 2)

        self.final_linear = nn.Sequential(nn.LayerNorm(dims[-1]))
        linear = nn.Linear(dims[-1], n_class)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), linear)

        self.apply(self.init_weights)
        self.set_dropout(None, drop_path)

    def set_dropout(self, dropout, drop_path):
        n_blocks = sum(self.depths)
        dp_rate = [drop_path * float(i) / n_blocks for i in range(n_blocks)]

        i = 0
        for block in self.block1:
            try:
                block.set_drop_path(dp_rate[i])
                i += 1

            except:
                continue

        for block in self.block2:
            try:
                block.set_drop_path(dp_rate[i])
                i += 1

            except:
                continue

        for block in self.block3:
            try:
                block.set_drop_path(dp_rate[i])
                i += 1

            except:
                continue

        for block in self.block4:
            try:
                block.set_drop_path(dp_rate[i])
                i += 1

            except:
                continue

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def make_block(
        self,
        depth,
        in_dim,
        dim,
        n_head,
        dim_head,
        dim_ff,
        window_size,
        reduction,
        drop_ff,
        drop_attn,
    ):
        block = [PatchEmbedding(in_dim, dim, reduction)]

        for i in range(depth):
            block.append(
                TransformerLayer(
                    dim,
                    n_head,
                    dim_head,
                    dim_ff,
                    window_size,
                    drop_ff=drop_ff,
                    drop_attn=drop_attn,
                )
            )

            if i == 0:
                block.append(PositionalEncodingGenerator(dim))

        return nn.Sequential(*block)

    def forward(self, input):
        out = self.block1(input.permute(0, 2, 3, 1))
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.final_linear(out).permute(0, 3, 1, 2)
        out = self.classifier(out)

        return out
