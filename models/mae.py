"""
Inspired from:
    - Kaiming He et al., “Masked Autoencoders Are Scalable Vision Learners,” arXiv, Nov. 2021, [Online]. Available: https://arxiv.org/abs/2111.06377.
    - GitHub page: https://github.com/facebookresearch/mae
"""

from collections import OrderedDict
from functools import partial
from typing import Any, Callable

import math
import torch
import torch.nn as nn

from .utils import program_conv_filters
from .activation import get_activation_class
from ssl.mae import TransformerBlock
from ssl.mae import get_sine_cosine_positional_embedding

__all__ = [
    "MaskedAutoencoderArtifact",
]


class MaskedAutoencoderArtifact(nn.Module):
    """MAE as per https://arxiv.org/abs/2111.06377."""

    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        in_channels: int,
        use_age: str,
        enc_num_heads: int,
        enc_dim: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_dim: int,
        dec_depth: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss: bool = False,
        art_filter_list: tuple = (9, 9, 9),
        art_dim: int = 64,
        art_dropout: float = 0.0,
        art_norm_layer: Callable[..., torch.nn.Module] = partial(nn.BatchNorm1d, eps=1e-6),
        art_use_age: bool = False,
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

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1
        elif self.use_age == "embedding":
            self.age_embedding = torch.nn.Parameter((torch.zeros(1, enc_dim, 1)))
            torch.nn.init.trunc_normal_(self.age_embedding, std=0.02)

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.activation = activation

        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_patches = seq_length // patch_size
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.dec_dim = dec_dim
        self.dec_depth = dec_depth
        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss

        self.art_dim = art_dim
        self.art_dropout = art_dropout
        self.art_norm_layer = art_norm_layer
        self.art_use_age = art_use_age

        ###########
        # Encoder #
        ###########
        # Linear projection
        self.enc_proj = nn.Conv1d(
            in_channels=in_channels,
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

        ###########
        # Decoder #
        ###########
        # Linear projection
        self.dec_proj = nn.Linear(enc_dim, dec_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim), requires_grad=False)

        # Positional embedding (sine-cosine)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dec_dim), requires_grad=False)

        # Decoder
        blk: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(enc_depth):
            blk[f"encoder_layer_{i}"] = TransformerBlock(
                dec_num_heads,
                dec_dim,
                round(dec_dim * mlp_ratio),
                dropout,
                attention_dropout,
                self.nn_act,
                norm_layer,
            )
        self.dec_blocks = nn.Sequential(blk)
        self.dec_norm = norm_layer(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, self.patch_size * in_channels, bias=True)  # decoder to patch

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
                    in_channels=art_dim if i > 0 else enc_dim,
                    out_channels=art_dim,
                    kernel_size=cf["kernel_size"],
                    padding=cf["kernel_size"] // 2,
                    stride=cf["stride"],
                    bias=True,
                ),
                self.art_norm_layer(),
                self.nn_act(),
            ]
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(self.art_dim, self.art_dim // 2, bias=False),
            nn.Dropout(p=self.art_dropout),
            self.art_norm_layer(),
            self.nn_act(),
            nn.Linear(self.art_dim // 2, 1, bias=True),
        ]
        self.art_net = nn.Sequential(*layers)

        # initialize params
        self.reset_weights()

    def reset_weights(self):
        # default reset
        self.apply(self._init_weights)

        # tokens
        torch.nn.init.normal_(self.class_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # positional embeddings (sine-cosine)
        self.enc_pos_embed.data.copy_(
            get_sine_cosine_positional_embedding(
                seq_len=self.n_patches, dim=self.enc_pos_embed.shape[-1], class_token=True
            )
            .float()
            .unsqueeze(0)
        )
        self.dec_pos_embed.data.copy_(
            get_sine_cosine_positional_embedding(
                seq_len=self.n_patches, dim=self.dec_pos_embed.shape[-1], class_token=True
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

    def random_masking(self, x, mask_ratio):
        N, l, D_e = x.size()
        l_keep = round(l * (1 - mask_ratio))

        # random sampling and sorting for masking
        random_noise = torch.rand(N, l, device=x.device)
        idx_shuffle = torch.argsort(random_noise, dim=1)
        idx_keep = idx_shuffle[:, :l_keep]

        # masking
        x_masked = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, D_e))
        mask = torch.ones((N, l), device=x.device)
        mask[:, :l_keep] = 0

        idx_restore = torch.argsort(idx_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=idx_restore)

        return x_masked, mask, idx_restore

    def forward_encoder(self, eeg, age, mask_ratio):
        # Reshape and permute the input tensor
        N, C, L = eeg.size()
        if self.use_age == "conv":
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            x = torch.cat((eeg, age), dim=1)

        # (N, C, L) -> (N, D_e, l)
        x = self.enc_proj(eeg)

        if self.use_age == "embedding":
            x = x + self.age_embedding * age.reshape(N, 1, 1)

        # (N, D_e, l) -> (N, l, D_e)
        # where N is the batch size, L is the source sequence length, and D is the embedding dimension
        x = x.permute(0, 2, 1)

        # positional encoding
        x = x + self.enc_pos_embed[:, 1:, :]

        # random masking
        x, mask, idx_restore = self.random_masking(x, mask_ratio)

        # class token
        # (N, l, D_e) -> (N, l + 1, D_e)
        class_token = self.class_token + self.enc_pos_embed[:, :1, :]
        batch_class_token = class_token.expand(N, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x.contiguous()

        # encoder stage
        x = self.enc_blocks(x)
        x = self.enc_norm(x)

        return x, mask, idx_restore

    def forward_decoder(self, x, idx_restore):
        # linear projection
        # (N, l + 1, D_e) -> (N, l + 1, D_d)
        x = self.dec_proj(x)

        # mask tokens
        # (N, l + 1, D_d) -> (N, l_full + 1, D_d)
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        mask_tokens = self.mask_token.repeat(x.shape[0], idx_restore.shape[1] - x.shape[1], 1)
        x = torch.cat((x, mask_tokens), dim=1)
        x = torch.gather(x, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat((cls_token, x), dim=1)

        # positional encoding
        x = x + self.dec_pos_embed

        # decoder stage
        x = self.dec_blocks(x)
        x = self.dec_norm(x)

        # predict
        # (N, l_full + 1, D_d) -> (N, l_full, p*C)
        x = self.decoder_pred(x[:, 1:, :])

        return x

    def patchify(self, eeg):
        N, C, L = eeg.size()
        p = self.patch_size
        l_full = L // p
        x = eeg.reshape(N, C, l_full, p)
        x = torch.einsum("NClp->NlpC", x)
        x = x.reshape(N, l_full, p * C)
        return x

    def unpatchify(self, x):
        N, l_full, D = x.size()
        p = self.patch_size
        C = D // p
        x = x.reshape(N, l_full, p, C)
        x = torch.einsum("NlpC->NClp", x)
        eeg = x.reshape(N, C, l_full * p)
        return eeg

    def compute_reconstruction_loss(self, eeg, pred, mask):
        # (N, C, L) -> (N, l_full, p*C)
        desired = self.patchify(eeg)
        if self.norm_pix_loss:
            mean = desired.mean(dim=-1, keepdim=True)
            var = desired.var(dim=-1, keepdim=True)
            desired = (desired - mean) / (var + 1e-6) ** 0.5

        # (N, l_full, p*C) -> (N, l_full)
        loss = (desired - pred) ** 2
        loss = loss.mean(dim=-1)
        # loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_artifact(self, eeg, mask):
        # Reshape the input tensor
        N, C, L = eeg.size()
        p = self.patch_size
        l_full = L // p
        x = eeg.reshape(N, C, l_full, p)
        x = torch.einsum("NClp->NlCp", x)
        x = x[mask]
        N, l_mask, C, p = x.size()
        x = x.reshape(N * l_mask, C, p)

        # apply small ConvNet to masked regions
        out = self.art_net(x)
        return out

    def forward(self, eeg: torch.Tensor, age: torch.Tensor, mask_ratio: float):
        # encoder
        latent, mask, idx_restore = self.forward_encoder(eeg, age, mask_ratio)

        # decoder
        pred = self.forward_decoder(latent, idx_restore)

        # reconstruction loss
        rec_loss = self.compute_reconstruction_loss(eeg, pred, mask)
        rec_loss = rec_loss[mask]

        # forward artifact
        art_out = self.forward_artifact(eeg, mask)

        # artifact loss
        art_loss = torch.nn.functional.smooth_l1_loss(art_out, rec_loss)

        return art_loss

    def post_update_params(self):
        pass
