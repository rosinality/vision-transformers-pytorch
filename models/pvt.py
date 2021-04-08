import math

import torch
from torch import nn
from torch.nn import functional as F

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
            self.reduce_norm = LayerNorm(dim)

    def forward(self, input, height, width, prev=None):
        batch_size = input.shape[0]

        def reshape(input):
            return input.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(
                1, 2
            )

        query = reshape(self.linear_q(input))

        if self.reduction > 1:
            dim = input.shape[-1]
            reduc = input.transpose(1, 2).reshape(batch_size, dim, height, width)
            reduc = self.reduce_conv(reduc).reshape(batch_size, dim, -1).transpose(1, 2)
            reduc = self.reduce_norm(reduc)
            kv = reduc

        else:
            kv = input

        key, value = self.linear_kv(kv).chunk(2, dim=2)
        key = reshape(key).transpose(2, 3)
        value = reshape(value)

        score = query @ key / math.sqrt(self.dim_head)

        if prev is not None:
            score = score + prev

        attn = F.softmax(score, 3)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = attn @ value

        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_head * self.n_head)
        out = self.linear(out)

        return out, score


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        dim_ff,
        activation=nn.SiLU,
        reduction=1,
        drop_ff=0,
        drop_attn=0,
        drop_path=0,
    ):
        super().__init__()

        self.norm_attn = LayerNorm(dim)
        self.attn = MultiHeadedAttention(dim, n_head, reduction, drop_attn)
        self.drop_path = DropPath(drop_path)
        self.norm_ff = LayerNorm(dim)
        self.ff = PositionwiseFeedForward(
            dim, dim_ff, activation=activation, dropout=drop_ff
        )

    def set_drop_path(self, p):
        self.drop_path.p = p

    def forward(self, input, height, width):
        out = input + self.drop_path(self.attn(self.norm_attn(input), height, width)[0])
        out = out + self.drop_path(self.ff(self.norm_ff(out)))

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, in_dim, dim, patch_size, cls_token=False, dropout=0):
        super().__init__()

        size = tuple2(patch_size)
        img_size = tuple2(image_size)

        self.conv = nn.Conv2d(in_dim, dim, size, stride=size)
        self.norm = LayerNorm(dim)

        height, width = img_size[0] // size[0], img_size[1] // size[1]
        n_patch = height * width

        if cls_token:
            n_patch += 1

        self.pos = nn.Parameter(torch.randn(n_patch, dim) * 0.02)

        self.cls_token = None
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(dim) * 0.02)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        batch, channel, height, width = input.shape

        out = self.conv(input)
        height, width = out.shape[2:]
        out = self.norm(out.flatten(2).transpose(1, 2))

        if self.cls_token is not None:
            cls_token = self.cls_token.view(1, 1, self.dim).expand(batch, -1, -1)
            out = torch.cat((cls_token, out), 1)

        out = out + self.pos.unsqueeze(0)
        out = self.dropout(out)

        return out, (height, width)


class PyramidVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        n_class,
        in_dim,
        depths,
        patch_embed_dims,
        n_heads,
        dim_ffs,
        reductions,
        drop_ff=0,
        drop_attn=0,
        drop_path=0,
    ):
        super().__init__()

        self.depths = depths

        self.patch_embedding = nn.ModuleList()
        patch_embed_dims = list(patch_embed_dims)
        cls_token = False
        patch_sizes = (4, 2, 2, 2)
        img_size = tuple2(image_size)
        for i, (p_in, p_out, p_size) in enumerate(
            zip([in_dim] + patch_embed_dims[:-1], patch_embed_dims, patch_sizes)
        ):
            if i == len(patch_embed_dims) - 1:
                cls_token = True

            self.patch_embedding.append(
                PatchEmbedding(
                    img_size, p_in, p_out, p_size, cls_token=cls_token, dropout=drop_ff
                )
            )
            img_size = (img_size[0] // p_size, img_size[1] // p_size)

        def make_block(i):
            return self.make_block(
                depths[i],
                patch_embed_dims[i],
                n_heads[i],
                dim_ffs[i],
                reductions[i],
                drop_ff,
                drop_attn,
            )

        self.block1 = make_block(0)
        self.block2 = make_block(1)
        self.block3 = make_block(2)
        self.block4 = make_block(3)

        self.norm = LayerNorm(patch_embed_dims[-1])
        self.classifier = nn.Linear(patch_embed_dims[-1], n_class)

        self.apply(self.init_weights)
        self.set_drop_path(drop_path)

    def set_drop_path(self, drop_path):
        depth = sum(self.depths)
        p = torch.linspace(0, drop_path, depth).tolist()

        i = 0

        for block in self.block1:
            block.set_drop_path(p[i])
            i += 1

        for block in self.block2:
            block.set_drop_path(p[i])
            i += 1

        for block in self.block3:
            block.set_drop_path(p[i])
            i += 1

        for block in self.block4:
            block.set_drop_path(p[i])
            i += 1

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def make_block(self, depth, dim, n_head, dim_ff, reduction, drop_ff, drop_attn):
        block = nn.ModuleList()

        for _ in range(depth):
            block.append(
                TransformerLayer(
                    dim,
                    n_head,
                    dim_ff,
                    reduction=reduction,
                    drop_ff=drop_ff,
                    drop_attn=drop_attn,
                )
            )

        return block

    def forward(self, input):
        batch = input.shape[0]

        out, (height, width) = self.patch_embedding[0](input)
        for block in self.block1:
            out = block(out, height, width)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        out, (height, width) = self.patch_embedding[1](out)
        for block in self.block2:
            out = block(out, height, width)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        out, (height, width) = self.patch_embedding[2](out)
        for block in self.block3:
            out = block(out, height, width)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        out, (height, width) = self.patch_embedding[3](out)
        for block in self.block4:
            out = block(out, height, width)

        out = self.norm(out[:, 0])
        out = self.classifier(out)

        return out
