import math

import torch
from torch import nn
from torch.nn import functional as F

from layer import DropPath, tuple2

LayerNorm = lambda x: nn.LayerNorm(x, eps=1e-6)


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, in_dim, dim=None, out_dim=None, activation=nn.SiLU, dropout=0):
        dim = in_dim if dim is None else dim
        out_dim = in_dim if out_dim is None else out_dim

        super().__init__(
            nn.Linear(in_dim, dim),
            activation(inplace=True),
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


class MultiHeadedHaloAttention(nn.Module):
    def __init__(self, dim, n_head, dim_head, window_size, halo_size, dropout=0):
        super().__init__()

        self.dim_head = dim_head
        self.n_head = n_head

        self.weight = nn.Linear(dim, n_head * dim_head * 3, bias=False)
        self.linear = nn.Linear(n_head * dim_head, dim)

        self.window_size = window_size
        self.halo_size = halo_size
        self.dropout = dropout

        rel_pos, max_pos = self.make_pos(window_size, halo_size)
        self.register_buffer("pos", rel_pos)
        self.rel_pos = nn.Embedding(max_pos + 1, n_head)
        self.rel_pos.weight.detach().zero_()

    def make_pos(self, window, halo):
        p = torch.arange(window + halo * 2)
        yy, xx = torch.meshgrid(p, p)
        x_diff = (xx.reshape(1, -1) - xx[halo:-halo, halo:-halo].reshape(-1, 1)) + (
            window + halo - 1
        )
        y_diff = (yy.reshape(1, -1) - yy[halo:-halo, halo:-halo].reshape(-1, 1)) + (
            window + halo - 1
        )
        pos = y_diff * (window + halo * 2) + x_diff
        max_pos = (window + halo - 1) * 2 * (window + halo * 2) + (
            window + halo - 1
        ) * 2

        return pos, max_pos

    def forward(self, input):
        batch, height, width, dim = input.shape
        h_stride = height // self.window_size
        w_stride = width // self.window_size
        window = self.window_size

        qkv = self.weight(input)
        query, keyvalue = qkv.split(
            (self.dim_head * self.n_head, self.dim_head * self.n_head * 2), -1
        )
        query = (
            query.view(
                batch, h_stride, window, w_stride, window, self.n_head, self.dim_head
            )
            .permute(0, 5, 1, 3, 2, 4, 6)
            .reshape(batch, self.n_head, -1, window ** 2, self.dim_head)
        )
        key, value = (
            F.unfold(
                keyvalue.permute(0, 3, 1, 2),
                window + self.halo_size * 2,
                stride=window,
                padding=self.halo_size,
            )
            .view(
                batch,
                2,
                self.n_head,
                self.dim_head,
                (window + self.halo_size * 2) ** 2,
                -1,
            )
            .permute(0, 1, 2, 5, 3, 4)
            .unbind(1)
        )
        value = value.transpose(-2, -1)

        score = query @ key / math.sqrt(self.dim_head)  # B, H, L, W^2, (W + A)^2
        rel_pos = self.rel_pos(self.pos)  # W, (W + A)^2, H
        score = score + rel_pos.view(1, 1, window ** 2, -1, self.n_head).permute(
            0, 4, 1, 2, 3
        )

        attn = F.softmax(score, -1)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ value  # B, H, L, W^2, D
        out = out.permute(0, 2, 3, 1, 4).reshape(
            batch, -1, window ** 2, self.n_head * self.dim_head
        )
        out = self.linear(out)  # B, L, W^2, D
        out = (
            out.view(batch, h_stride, w_stride, window, window, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, -1)
        )

        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        dim_head,
        dim_ff,
        window_size,
        halo_size,
        activation=nn.SiLU,
        drop_ff=0,
        drop_attn=0,
        drop_path=0,
    ):
        super().__init__()

        self.norm_attn = LayerNorm(dim)
        self.attn = MultiHeadedHaloAttention(
            dim, n_head, dim_head, window_size, halo_size, drop_attn
        )
        self.drop_path = DropPath(drop_path)
        self.norm_ff = LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=activation, dropout=drop_ff
        )

    def set_drop_path(self, p):
        self.drop_path.p = p

    def forward(self, input):
        input += self.drop_path(self.attn(self.norm_attn(input)))
        input += self.drop_path(self.ff(self.norm_ff(input)))

        return input


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


class HaloTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        n_class,
        depths,
        dims,
        dim_head,
        n_heads,
        dim_ffs,
        window_size,
        halo_size,
        drop_ff=0,
        drop_attn=0,
        drop_path=0,
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
                halo_size,
                reduction,
                drop_ff,
                drop_attn,
                drop_path,
            )

        self.block1 = make_block(0, 3, 4)
        self.block2 = make_block(1, dims[0], 2)
        self.block3 = make_block(2, dims[1], 2)
        self.block4 = make_block(3, dims[2], 2)

        self.final_linear = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], dims[-1] * 2),
            nn.LayerNorm(dims[-1] * 2),
            nn.SiLU(inplace=True),
        )
        linear = nn.Linear(dims[-1] * 2, n_class)
        nn.init.normal_(linear.weight, std=0.01)
        nn.init.zeros_(linear.bias)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), linear)

        self.apply(self.init_weights)

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
        halo_size,
        reduction,
        drop_ff,
        drop_attn,
        drop_path,
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
                    halo_size,
                    drop_ff=drop_ff,
                    drop_attn=drop_attn,
                    drop_path=drop_path,
                )
            )

        return nn.Sequential(*block)

    def forward(self, input):
        out = self.block1(input.permute(0, 2, 3, 1))
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.final_linear(out).permute(0, 3, 1, 2)
        out = self.classifier(out)

        return out
