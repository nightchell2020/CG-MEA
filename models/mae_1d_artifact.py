"""
Inspired from:
    - Kaiming He et al., “Masked Autoencoders Are Scalable Vision Learners,” arXiv, Nov. 2021, [Online]. Available: https://arxiv.org/abs/2111.06377.
    - GitHub page: https://github.com/facebookresearch/mae
"""

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Union

import math
import torch
import torch.nn as nn

from .utils import program_conv_filters
from .activation import get_activation_class
from .ssl.mae_1d import TransformerBlock
from .ssl.mae_1d import get_sine_cosine_positional_embedding

__all__ = [
    "MaskedAutoencoder1DArtifact",
    "mae_1d_art_b_e768_d512",
    "mae_1d_art_l_e1024_d512",
    "mae_1d_art_h_e1280_d512",
]


class MaskedAutoencoder1DArtifact(nn.Module):
    """MAE as per https://arxiv.org/abs/2111.06377."""

    def __init__(
        self,
        seq_length: int,
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
        head_norm_layer: Callable[..., torch.nn.Module] = partial(nn.BatchNorm1d, eps=1e-6),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        art_filter_list: tuple = (9, 9, 9),
        art_dim: int = 64,
        art_dropout: float = 0.0,
        art_norm_layer: Callable[..., torch.nn.Module] = partial(nn.BatchNorm1d, eps=1e-6),
        art_use_age: str = "no",
        art_out_activation: str = "none",
        global_pool: bool = True,
        descending: Union[bool, str] = False,
        **kwargs: Any,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "embedding", "no"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(use_age) receives one of ['fc', 'conv', 'embedding', 'no']."
            )
        if seq_length % patch_size > 0:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(seq_length, patch_size) requires seq_length to "
                f"be multiple of patch_size."
            )
        if art_use_age not in ["conv", "embedding", "no"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(art_use_age) receives one of ['conv', 'embedding', 'no']."
            )
        if art_out_activation not in ["none", "relu", "softplus"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(art_out_activation) receives one of ['none', 'relu', 'softplus']."
            )

        self.use_age = use_age
        if self.use_age == "embedding":
            self.age_embed = torch.nn.Parameter((torch.zeros(1, enc_dim, 1)))
            torch.nn.init.trunc_normal_(self.age_embed, std=0.02)

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.activation = activation

        self.seq_length = seq_length
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.n_patches = seq_length // patch_size
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer

        self.art_dim = art_dim
        self.art_dropout = art_dropout
        self.art_norm_layer = art_norm_layer
        self.art_out_nn_act = get_activation_class(art_out_activation, class_name=self.__class__.__name__)
        self.art_use_age = art_use_age
        if self.art_use_age == "embedding":
            self.art_age_embed = torch.nn.Parameter((torch.zeros(1, in_channels, 1)))
            torch.nn.init.trunc_normal_(self.art_age_embed, std=0.02)

        self.global_pool = global_pool
        self.head_norm_layer = head_norm_layer
        self.fc_stages = fc_stages
        self.out_dims = out_dims
        self.descending = descending

        ###########
        # Encoder #
        ###########
        # Linear projection
        self.enc_proj = nn.Conv1d(
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

        ######################
        # Artifact estimator #
        ######################
        conv_filter_list = [{"kernel_size": k} for k in art_filter_list]
        program_conv_filters(
            sequence_length=self.patch_size,
            conv_filter_list=conv_filter_list,
            output_lower_bound=2,
            output_upper_bound=5,
            class_name=self.__class__.__name__,
        )
        layers = []
        for i, cf in enumerate(conv_filter_list):
            layers += [
                nn.Conv1d(
                    in_channels=art_dim if i > 0 else in_channels if self.art_use_age != "conv" else in_channels + 1,
                    out_channels=art_dim,
                    kernel_size=cf["kernel_size"],
                    padding=cf["kernel_size"] // 2,
                    stride=cf["stride"],
                    bias=True,
                ),
                self.art_norm_layer(art_dim),
                self.nn_act(),
            ]
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(self.art_dim, self.art_dim // 2, bias=False),
            nn.Dropout(p=self.art_dropout),
            self.art_norm_layer(self.art_dim // 2),
            self.nn_act(),
            nn.Linear(self.art_dim // 2, 1, bias=True),
            self.art_out_nn_act(),
        ]
        self.art_net = nn.Sequential(*layers)

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
        torch.nn.init.normal_(self.class_token, std=0.02)

        # positional embeddings (sine-cosine)
        self.enc_pos_embed.data.copy_(
            get_sine_cosine_positional_embedding(
                seq_len=self.n_patches, dim=self.enc_pos_embed.shape[-1], class_token=True
            )
            .float()
            .unsqueeze(0)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            fan_in = m.in_channels * m.kernel_size[0]
            nn.init.trunc_normal_(m.weight, std=math.sqrt(1 / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def artifact_masking(self, x, art_out, mask_ratio):
        N, l_full, D_e = x.size()
        l_keep = round(l_full * (1 - mask_ratio))

        if isinstance(self.descending, bool):
            idx_rank = torch.argsort(art_out, dim=1, descending=self.descending)
            idx_keep = idx_rank[:, :l_keep]
        elif self.descending == "both":
            idx_rank = torch.argsort(art_out, dim=1)
            l_discard = (l_full - l_keep) // 2
            idx_keep = idx_rank[:, l_discard:-l_discard]
        else:
            raise ValueError(
                f"{self.__class__.__name__}.artifact_masking() has " f"uninterpretable self.descending value."
            )
        # masking
        # (N, l_full, D_e) -> (N, l, D_e)
        x_masked = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, D_e))
        mask = torch.ones((N, l_full), device=x.device)
        mask[:, :l_keep] = 0

        idx_restore = torch.argsort(idx_rank, dim=1)
        mask = torch.gather(mask, dim=1, index=idx_restore)

        return x_masked, mask, idx_restore

    def forward_encoder(self, eeg, age, art_out, mask_ratio):
        # Reshape and permute the input tensor
        N, C, L = eeg.size()
        if self.use_age == "conv":
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            eeg = torch.cat((eeg, age), dim=1)

        # (N, C, L) -> (N, D_e, l_full)
        x = self.enc_proj(eeg)

        if self.use_age == "embedding":
            x = x + self.age_embed * age.reshape(N, 1, 1)

        # (N, D_e, l_full) -> (N, l_full, D_e)
        # where N is the batch size, L is the source sequence length, and D is the embedding dimension
        x = x.permute(0, 2, 1)

        # positional encoding
        x = x + self.enc_pos_embed[:, 1:, :]

        # masking according to the predicted loss from artifact network
        # (N, l_full, D_e) -> (N, l, D_e)
        x, mask, idx_restore = self.artifact_masking(x, art_out, mask_ratio)

        # class token
        # (N, l, D_e) -> (N, l + 1, D_e)
        class_token = self.class_token + self.enc_pos_embed[:, :1, :]
        batch_class_token = class_token.expand(N, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x.contiguous()

        # encoder stage
        x = self.enc_blocks(x)
        # x = self.enc_norm(x)

        return x, mask, idx_restore

    def patchify(self, eeg):
        N, C, L = eeg.size()
        p = self.patch_size
        l_full = L // p
        x = eeg.reshape(N, C, l_full, p)
        x = torch.einsum("NClp->NlpC", x)
        x = x.reshape(N, l_full, p * C)
        return x

    def forward_artifact(self, eeg, age):
        # Reshape the input tensor
        N, C, L = eeg.size()
        p = self.patch_size
        l_full = L // p

        if self.art_use_age == "conv":
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            eeg = torch.cat((eeg, age), dim=1)
            C = C + 1
        elif self.art_use_age == "embedding":
            eeg = eeg + self.art_age_embed * age.reshape(N, 1, 1)

        # (N, C, L) -> (N, l_full, C, p)
        x = eeg.reshape(N, C, l_full, p)
        x = torch.einsum("NClp->NlCp", x)

        # (N, l_full, C, p) -> (N*l_full, C, p)
        x = x.reshape(N * l_full, C, p)

        # apply small ConvNet to masked regions
        out = self.art_net(x)
        out = out.squeeze().reshape(N, l_full)
        return out

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

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        # artifact
        art_out = self.forward_artifact(x, age)

        # encoder
        x, mask, idx_restore = self.forward_encoder(x, age, art_out, self.mask_ratio)

        # head
        out = self.forward_head(x, age, target_from_last)
        return out

    def forward(self, eeg: torch.Tensor, age: torch.Tensor, target_from_last: int = 0):
        out = self.compute_feature_embedding(eeg, age, target_from_last)
        return out

    def finetune_mode(self, mode: str = "finetune"):
        mode = mode.lower()
        if mode not in ["finetune", "fc_stage", "from_scratch"]:
            raise ValueError(
                f"{self.__class__.__name__}.tuning_mode(mode) receives one of ['finetune', 'fc_stage', 'from_scratch']."
            )

        if mode == "fc_stage":
            self.requires_grad_(False)
            self.eval()
            self.fc_stage.requires_grad_(True)
            self.fc_stage.train()
        elif mode == "finetune":
            self.requires_grad_(True)
            self.train()
            self.art_net.requires_grad_(False)
            self.art_net.eval()
            self.enc_pos_embed.requires_grad_(False)
            for k, v in self._parameters.items():
                if k.startswith("art"):
                    v.requires_grad_(False)
        elif mode == "from_scratch":
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

            if n in ["class_token", "enc_pos_embed", "age_embed", "art_age_embed"]:
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

        # import json
        # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

        return list(param_groups.values())


def mae_1d_art_b_e768_d512(**kwargs):
    model = MaskedAutoencoder1DArtifact(
        arch="mae_1d_b_e768_d512",
        enc_num_heads=12,
        enc_dim=768,
        enc_depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_1d_art_l_e1024_d512(**kwargs):
    model = MaskedAutoencoder1DArtifact(
        arch="mae_1d_l_e1024_d512",
        enc_num_heads=16,
        enc_dim=1024,
        enc_depth=24,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_1d_art_h_e1280_d512(**kwargs):
    model = MaskedAutoencoder1DArtifact(
        arch="mae_1d_h_e1280_d512",
        enc_num_heads=16,
        enc_dim=1280,
        enc_depth=32,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
