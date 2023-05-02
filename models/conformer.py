"""
Modified from:
    - lucidrains's implementation of Conformer architecture
    - https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
    - Conformer paper: https://arxiv.org/abs/2005.08100
"""
from typing import Tuple
from collections import OrderedDict
import math

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from .activation import get_activation_class


# helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


# helper classes
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module
class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, = x.shape[-2], x.device, self.heads
        max_pos_emb, has_context = self.max_pos_emb, exists(context)
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = default(context_mask, mask) if not has_context \
                else default(context_mask, lambda: torch.ones(*context.shape[:2], device=device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4, conv_expansion_factor=2,
                 conv_kernel_size=31, attn_dropout=0.0, ff_dropout=0.0, conv_dropout=0.0):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class ConformerClassifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_dims: int,
                 seq_len_2d: Tuple[int],
                 use_age: str,
                 fc_stages: int,
                 encoder_dim: int = 512,
                 num_layers: int = 17,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 final_pool: str = 'average',
                 **kwargs):
        super().__init__()

        if use_age not in ['fc', 'conv', 'embedding', 'no']:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) "
                             f"receives one of ['fc', 'conv', 'embedding', 'no'].")

        if final_pool not in ['average', 'max']:
            raise ValueError(f"{self.__class__.__name__}.__init__(final_pool) both "
                             f"receives one of ['average', 'max'].")

        if fc_stages < 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(fc_stages) receives "
                             f"an integer equal to ore more than 1.")

        self.use_age = use_age
        if self.use_age == 'conv':
            in_channels += 1
        elif self.use_age == 'embedding':
            self.age_embedding = torch.nn.Parameter((torch.zeros(1, 1, encoder_dim)))
            torch.nn.init.trunc_normal_(self.age_embedding, std=.02)

        self.base_channels = encoder_dim
        self.fc_stages = fc_stages
        self.seq_len_2d = seq_len_2d
        self.dropout = dropout
        self.num_classes = out_dims
        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.activation = activation

        self.conv_subsample = nn.Sequential(
                nn.Conv2d(in_channels, encoder_dim, kernel_size=3, stride=2),
                self.nn_act(),
                nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=2),
                self.nn_act(),
            )
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim * (((seq_len_2d[0] - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=dropout),
        )

        self.output_length = ((seq_len_2d[1] - 1) // 2 - 1) // 2

        self.conformer_layers = nn.ModuleList([ConformerBlock(
            dim=encoder_dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1
        ) for _ in range(num_layers)])

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        prev_dim = encoder_dim
        if self.use_age == 'fc':
            prev_dim = prev_dim + 1
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(self.fc_stages - 1):
            # TODO: Dropout or Normalization layers can be added here.
            heads_layers[f"linear{i + 1}"] = nn.Linear(prev_dim, prev_dim // 2)
            heads_layers[f"dropout{i + 1}"] = nn.Dropout(dropout)
            heads_layers[f"act{i + 1}"] = self.nn_act()
            prev_dim = prev_dim // 2
        heads_layers["head"] = nn.Linear(prev_dim, out_dims)
        self.heads = nn.Sequential(heads_layers)
        # self.fc_stage = self.heads

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        for m in self.conv_subsample.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for i in range(self.fc_stages - 1):
            linear_name = f"linear{i + 1}"
            if hasattr(self.heads, linear_name) and isinstance(getattr(self.heads, linear_name), nn.Linear):
                fan_in = getattr(self.heads, linear_name).in_features
                if self.activation == 'tanh':
                    nn.init.trunc_normal_(getattr(self.heads, linear_name).weight, std=math.sqrt(1 / fan_in))
                else:
                    nn.init.trunc_normal_(getattr(self.heads, linear_name).weight, std=math.sqrt(2.0 / fan_in))
                nn.init.zeros_(getattr(self.heads, linear_name).bias)
        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def get_output_length(self):
        return self.output_length

    def get_dims_from_last(self, target_from_last: int):
        l = self.fc_stages - target_from_last
        return self.heads[l].in_features

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        n, _, h, w = x.size()
        if self.use_age == 'conv':
            age = age.reshape((n, 1, 1, 1)).expand(n, 1, h, w)
            x = torch.cat((x, age), dim=1)
        x = x.permute(0, 1, 3, 2)  # N, C, F, T -> N, C, T, F

        # subsample stage
        x = self.conv_subsample(x)
        x = x.permute(0, 2, 1, 3)  # N, C', T, F -> N, T, C', F
        n, t, c, f = x.shape
        x = x.contiguous().view(n, t, c * f)  # N, T, C', F -> N, T, (C' * F)

        # linear and dropout
        x = self.input_projection(x)  # N, T, (C' * F) -> N, T, D

        if self.use_age == 'embedding':
            x = x + self.age_embedding * age.reshape(n, 1, 1)

        # conformer stages
        for conformer_layer in self.conformer_layers:
            x = conformer_layer(x)

        # fc stages
        x = x.permute(0, 2, 1)  # N, T, D -> N, D, T
        x = self.final_pool(x)  # N, D, T -> N, D, 1
        x = torch.flatten(x, 1)

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        if target_from_last == 0:
            x = self.heads(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                                 f"an integer equal to or smaller than fc_stages={self.fc_stages}.")

            for l in range(self.fc_stages - target_from_last):
                x = self.heads[l](x)
        return x

    def forward(self, x: torch.Tensor, age: torch.Tensor):
        x = self.compute_feature_embedding(x, age)
        # return F.log_softmax(x, dim=1)
        return x