import math
from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tensorfn.config import config_model
from pydantic import StrictInt, StrictFloat, StrictBool

from .layer import DropPath, PositionwiseFeedForward, tuple2


LayerNorm = lambda x: nn.LayerNorm(x, eps=1e-6)


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, n_head, bias=True, dropout=0):
        super().__init__()

        self.dim_head = dim // n_head
        self.n_head = n_head

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dim)

    def forward(self, input):
        batch, length, dim = input.shape

        qkv = (
            self.qkv(input)
            .reshape(batch, length, 3, self.n_head, self.dim_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = self.linear(out)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_head, dim_ff, dropout, drop_attn, drop_ff, drop_path):
        super().__init__()

        self.norm_attn = LayerNorm(dim)
        self.attn = MultiHeadedAttention(dim, n_head, dropout=drop_attn)
        self.norm_ff = LayerNorm(dim)
        self.ff = PositionwiseFeedForward(dim, dim_ff, dropout=drop_ff)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        out = input + self.drop_path(self.dropout(self.attn(self.norm_attn(input))))
        out = out + self.drop_path(self.dropout(self.ff(self.norm_ff(out))))

        return out

    def set_drop_path(self, p):
        self.drop_path.p = p


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, window_size):
        super().__init__()

        self.linear = nn.Conv2d(in_dim, out_dim, window_size, stride=window_size)

    def forward(self, input):
        return self.linear(input).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        head,
        image_size,
        window_size,
        depth,
        dim,
        n_head,
        dim_ff,
        dropout,
        drop_attn,
        drop_ff,
        drop_path,
    ):
        super().__init__()

        image_size = tuple2(image_size)
        n_patch = (image_size[0] // window_size) * (image_size[1] // window_size)

        self.patch_embedding = PatchEmbedding(3, dim, window_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        drop_path_rate = torch.linspace(0, drop_path, depth).tolist()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(dim, n_head, dim_ff, dropout, drop_attn, drop_ff, dpr)
                for dpr in drop_path_rate
            ]
        )

        self.norm = LayerNorm(dim)

        self.apply(self.init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        self.head = head
        self.depth = depth

    def set_drop_path(self, drop_path):
        drop_path_rate = torch.linspace(0, drop_path, self.depth).tolist()

        for layer, p in zip(self.layers, drop_path_rate):
            layer.set_drop_path(p)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_feature(self, input):
        out = self.patch_embedding(input)
        out = torch.cat((self.cls_token.expand(out.shape[0], -1, -1), out), 1)
        pos_embed = self.interpolate_pos_embedding(out, self.pos_embed)
        out = out + pos_embed
        out = self.pos_drop(out)

        for layer in self.layers:
            out = layer(out)

        out = self.norm(out)

        return out[:, 0]

    def interpolate_pos_embedding(self, input, pos_embed):
        n_patch = input.shape[1] - 1
        n_pos = pos_embed.shape[1] - 1

        if n_patch == n_pos:
            return pos_embed

        cls_embed = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = input.shape[-1]

        pos_embed = F.interpolate(
            pos_embed.reshape(
                1, int(math.sqrt(n_pos)), int(math.sqrt(n_pos)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(n_patch / n_pos),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((cls_embed.unsqueeze(0), pos_embed), 1)

    def forward(self, input):
        if not isinstance(input, (list, tuple)):
            input = [input]

        crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([i.shape[-1] for i in input]), return_counts=True
            )[1],
            0,
        )
        start = 0

        for end in crops:
            out = self.forward_feature(torch.cat(input[start:end]))

            if start == 0:
                output = out

            else:
                output = torch.cat((output, out))

            start = end

        if self.head is not None:
            output = self.head(output)

        return output


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        depth=3,
        dim_ff=2048,
        dim_bottleneck=256,
    ):
        super().__init__()

        if depth == 1:
            self.mlp = nn.Linear(in_dim, dim_bottleneck)

        else:
            layers = [nn.Linear(in_dim, dim_ff)]

            if use_bn:
                layers.append(nn.BatchNorm1d(dim_ff))

            layers.append(nn.GELU())

            for _ in range(depth - 2):
                layers.append(nn.Linear(dim_ff, dim_ff))

                if use_bn:
                    layers.append(nn.BatchNorm1d(dim_ff))

                layers.append(nn.GELU())

            layers.append(nn.Linear(dim_ff, dim_bottleneck))

            self.mlp = nn.Sequential(*layers)

        self.apply(self.init_weights)

        self.last = nn.utils.weight_norm(nn.Linear(dim_bottleneck, out_dim, bias=False))
        self.last.weight_g.detach().fill_(1)

        if norm_last_layer:
            self.last.weight_g.requires_grad = False

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input):
        out = self.mlp(input)
        out = F.normalize(out, dim=-1, p=2)
        out = self.last(out)

        return out


@config_model(name="dino", namespace="model", use_type=True)
def dino(
    image_size: Union[StrictInt, Tuple[StrictInt, StrictInt]],
    window_size: StrictInt,
    depth: StrictInt,
    dim: StrictInt,
    n_head: StrictInt,
    dim_ff: StrictInt,
    dropout: StrictFloat,
    drop_attn: StrictFloat,
    drop_ff: StrictFloat,
    drop_path: StrictFloat,
    dim_head_out: StrictInt,
    use_bn: StrictBool = False,
    norm_last_layer: StrictBool = True,
    depth_head: StrictInt = 3,
    dim_head_ff: StrictInt = 2048,
    dim_head_bottleneck: StrictInt = 256,
):
    head = DINOHead(
        dim,
        dim_head_out,
        use_bn,
        norm_last_layer,
        depth_head,
        dim_head_ff,
        dim_head_bottleneck,
    )
    model = VisionTransformer(
        head,
        image_size,
        window_size,
        depth,
        dim,
        n_head,
        dim_ff,
        dropout,
        drop_attn,
        drop_ff,
        drop_path,
    )

    return model
