"""
Inspired from:
    - Kaiming He et al., “Masked AutoEncoders Are Scalable Vision Learners,” arXiv, Nov. 2021, [Online]. Available: https://arxiv.org/abs/2111.06377.
    - GitHub page: https://github.com/facebookresearch/mae
"""

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Tuple
import warnings

import math
import torch
import torch.nn as nn

from .activation import get_activation_class
from .ssl.mae_1d import TransformerBlock
from .ssl.mae_2d import get_2d_sine_cosine_positional_embedding

__all__ = [
    "MaskedAutoencoder2D",
    "mae_2d_b_e768_d512",
    "mae_2d_l_e1024_d512",
    "mae_2d_h_e1280_d512",
]


class MaskedAutoencoder2D(nn.Module):
    """MAE as per https://arxiv.org/abs/2111.06377."""

    def __init__(
        self,
        seq_len_2d: Tuple[int],
        patch_size: int,
        mask_ratio: float,
        in_channels: int,
        use_age: str,
        enc_num_heads: int,
        enc_dim: int,
        enc_depth: int,
        mlp_ratio: float,
        fc_stages: int,
        out_dims: int,
        head_norm_layer: Callable[..., nn.Module] = partial(nn.BatchNorm1d, eps=1e-6),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        global_pool: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "embedding", "no"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(use_age) receives one of ['fc', 'conv', 'embedding', 'no']."
            )
        if seq_len_2d[0] % patch_size > 0 or seq_len_2d[1] % patch_size > 0:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(seq_len_2d, patch_size) requires seq_len_2d[0] and "
                f"seq_len_2d[1] to be multiple of patch_size: {seq_len_2d, patch_size}"
            )

        self.use_age = use_age
        if self.use_age == "embedding":
            self.age_embed = nn.Parameter((torch.zeros(1, enc_dim, 1)))
            nn.init.trunc_normal_(self.age_embed, std=0.02)

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.activation = activation

        self.seq_len_2d = seq_len_2d
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.n_patches = (seq_len_2d[0] // patch_size) * (seq_len_2d[1] // patch_size)
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer

        self.global_pool = global_pool
        self.head_norm_layer = head_norm_layer
        self.fc_stages = fc_stages
        self.out_dims = out_dims

        ###########
        # Encoder #
        ###########
        # Linear projection
        self.enc_proj = nn.Conv2d(
            in_channels=in_channels if self.use_age != "conv" else in_channels + 1,
            out_channels=enc_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        # Class token
        self.output_length = self.n_patches + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, enc_dim))

        # Positional embedding (sine-cosine)
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, enc_dim), requires_grad=False)

        # Encoder
        blk: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(enc_depth):
            blk[f"encoder_layer_{i}"] = TransformerBlock(
                enc_num_heads,
                enc_dim,
                round(enc_dim * mlp_ratio),
                dropout,
                attention_dropout,
                self.nn_act,
                norm_layer,
            )
        self.enc_blocks = nn.Sequential(blk)
        self.enc_norm = norm_layer(enc_dim)

        #######################
        # Classification head #
        #######################
        fc_stage = []
        current_dims = self.enc_dim
        if self.use_age == "fc":
            current_dims = current_dims + 1

        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(current_dims, current_dims // 2, bias=False),
                nn.Dropout(p=dropout),
                head_norm_layer(current_dims // 2),
                self.nn_act(),
            )
            current_dims = current_dims // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(current_dims, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

        # initialize params
        self.reset_weights()

    def reset_weights(self):
        # default reset
        self.apply(self._init_weights)

        # tokens
        nn.init.normal_(self.class_token, std=0.02)

        # positional embeddings (sine-cosine)
        self.enc_pos_embed.data.copy_(
            get_2d_sine_cosine_positional_embedding(
                seq_len_2d=self.seq_len_2d,
                patch_size=self.patch_size,
                dim=self.enc_pos_embed.shape[-1],
                class_token=True,
            )
            .float()
            .unsqueeze(0)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def forward_encoder(self, img, age):
        # Reshape and permute the input tensor
        N, C, H, W = img.size()
        if self.use_age == "conv":
            age = age.reshape((N, 1, 1, 1)).expand(N, 1, H, W)
            img = torch.cat((img, age), dim=1)

        # (N, C, H, W) -> (N, D_e, h, w) -> (N, D_e, l_full)
        x = self.enc_proj(img)
        x = x.reshape(N, self.enc_dim, -1)

        if self.use_age == "embedding":
            x = x + self.age_embed * age.reshape(N, 1, 1)

        # (N, D_e, l_full) -> (N, l_full, D_e)
        # where N is the batch size, l_full is the source sequence length, and D_e is the embedding dimension
        x = x.permute(0, 2, 1)

        # positional encoding
        x = x + self.enc_pos_embed[:, 1:, :]

        # class token
        # (N, l, D_e) -> (N, l + 1, D_e)
        class_token = self.class_token + self.enc_pos_embed[:, :1, :]
        batch_class_token = class_token.expand(N, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x.contiguous()

        # encoder stage
        x = self.enc_blocks(x)
        # x = self.enc_norm(x)

        return x

    def patchify(self, img):
        N, C, H, W = img.size()
        p = self.patch_size

        if H != self.seq_len_2d[0]:
            raise ValueError(
                f"{self.__class__.__name__}.patchify(img) img[0]: {H} have to be the same as "
                f"seq_len_2d[0]: {self.seq_len_2d[0]}."
            )
        if H % p != 0:
            warnings.warn(
                f"{self.__class__.__name__}.patchify(img) img[0]: {H} is not multiple of patch_size: {p}.",
                UserWarning,
            )
        elif W % p != 0:
            warnings.warn(
                f"{self.__class__.__name__}.patchify(img) img[1]: {W} is not multiple of patch_size: {p}.",
                UserWarning,
            )

        h = H // p
        w = W // p
        x = img.reshape(N, C, h, p, w, p)
        x = torch.einsum("NChpwq->NhwpqC", x)
        x = x.reshape(N, h * w, p * p * C)
        return x

    def forward_head(self, x, age, target_from_last: int = 0):
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            x = self.enc_norm(x)
            x = x[:, 0]

        if self.use_age == "fc":
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        if target_from_last == 0:
            x = self.fc_stage(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.forward_head(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.fc_stage[l](x)

        return x

    def compute_feature_embedding(self, img, age, target_from_last: int = 0):
        # encoder
        x = self.forward_encoder(img, age)

        # head
        out = self.forward_head(x, age, target_from_last)
        return out

    def forward(self, img: torch.Tensor, age: torch.Tensor, target_from_last: int = 0):
        out = self.compute_feature_embedding(img, age, target_from_last)
        return out

    def finetune_mode(self, mode: str = "finetune"):
        mode = mode.lower()
        if mode not in ["finetune", "fc_stage"]:
            raise ValueError(f"{self.__class__.__name__}.tuning_mode(mode) receives one of ['finetune', 'fc_stage'].")

        if mode == "fc_stage":
            self.requires_grad_(False)
            self.eval()
            self.fc_stage.requires_grad_(True)
            self.fc_stage.train()
        elif mode == "finetune":
            self.requires_grad_(True)
            self.train()
            self.enc_pos_embed.requires_grad_(False)

    def layer_wise_lr_params(self, weight_decay=0.05, layer_decay=0.75):
        """
        Parameter groups for layer-wise lr decay
        Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
        """
        param_group_names = {}
        param_groups = {}

        num_layers = len(self.enc_blocks) + 1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            # no decay: all 1D parameters and model specific ones
            if p.ndim == 1:
                g_decay = "no_decay"
                this_decay = 0.0
            else:
                g_decay = "decay"
                this_decay = weight_decay

            if n in ["class_token", "enc_pos_embed", "age_embed"]:
                layer_id = 0
            elif n.startswith("enc_proj"):
                layer_id = 0
            elif "blocks" in n:
                layer_id = int(n.split(".")[1].split("_")[-1]) + 1
            else:
                layer_id = num_layers

            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

        # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

        return list(param_groups.values())


def mae_2d_b_e768_d512(**kwargs):
    model = MaskedAutoencoder2D(
        arch="mae_2d_b_e768_d512",
        enc_num_heads=12,
        enc_dim=768,
        enc_depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_2d_l_e1024_d512(**kwargs):
    model = MaskedAutoencoder2D(
        arch="mae_2d_l_e1024_d512",
        enc_num_heads=16,
        enc_dim=1024,
        enc_depth=24,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_2d_h_e1280_d512(**kwargs):
    model = MaskedAutoencoder2D(
        arch="mae_2d_h_e1280_d512",
        enc_num_heads=16,
        enc_dim=1280,
        enc_depth=32,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
