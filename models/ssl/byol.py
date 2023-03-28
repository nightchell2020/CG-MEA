"""
Inspired from:
    - J.-B. Grill et al., “Bootstrap your own latent: A new approach to self-supervised Learning,” arXiv, vol. 200, Jun. 2020,
    [Online]. Available: http://arxiv.org/abs/2006.07733.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..activation import get_activation_class


class BYOL(nn.Module):
    def __init__(self,
                 backbone,
                 embedding_layer,
                 activation='relu',
                 mlp_hidden_size=4096,
                 projection_size=256,
                 target_ema=0.99,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.online_backbone = self.backbone
        self.embedding_layer = embedding_layer
        self.base_dim = backbone.get_dims_from_last(embedding_layer)
        self.cut_last_dim = self.online_backbone.use_age == 'fc' and \
                            self.embedding_layer == self.online_backbone.get_num_fc_stages()
        if self.cut_last_dim:
            self.base_dim += -1
        self.mlp_hidden_size = mlp_hidden_size
        self.projection_size = projection_size
        self.target_ema = target_ema

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        self.online_proj = nn.Sequential(nn.Linear(self.base_dim, self.mlp_hidden_size),
                                         nn.BatchNorm1d(self.mlp_hidden_size),
                                         self.nn_act(),
                                         nn.Linear(self.mlp_hidden_size, self.projection_size))
        self.online_pred = nn.Sequential(nn.Linear(self.projection_size, self.mlp_hidden_size),
                                         nn.BatchNorm1d(self.mlp_hidden_size),
                                         self.nn_act(),
                                         nn.Linear(self.mlp_hidden_size, self.projection_size))

        self.target_backbone = deepcopy(self.online_backbone)
        self.target_backbone.requires_grad_(False)
        self.target_proj = deepcopy(self.online_proj)
        self.target_proj.requires_grad_(False)

    def post_update_params(self):
        # backbone
        for online_p, target_p in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            target_p.data = self.target_ema * target_p.data + (1 - self.target_ema) * online_p.data
        # projection head
        for online_p, target_p in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            target_p.data = self.target_ema * target_p.data + (1 - self.target_ema) * online_p.data

    def _compute_embedding(self, x, age, network):
        out_a_online = network.compute_feature_embedding(x, age, self.embedding_layer)
        if self.cut_last_dim:
            return out_a_online[:, :-1]
        else:
            return out_a_online
    def forward(self, x: torch.Tensor, age: torch.Tensor):
        # divide x's depending on the branch
        N = x.shape[0]
        if N % 2 != 0:
            raise ValueError(f"{self.__class__.__name__}.forward(x, age) minibatch size is not multiple of 2.")

        x_a, age_a = x[::2], age[::2]
        x_b, age_b = x[1::2], age[1::2]

        # run online and target network and compute loss
        out_a_online = self._compute_embedding(x_a, age_a, self.online_backbone)
        out_a_online = self.online_pred(self.online_proj(out_a_online))
        out_a_online = F.normalize(out_a_online, p=2, dim=-1)
        with torch.no_grad():
            out_b_target = self._compute_embedding(x_b, age_b, self.target_backbone)
            out_b_target = self.target_proj(out_b_target.detach())  # with stop_gradient and no prediction
            out_b_target = F.normalize(out_b_target, p=2, dim=-1)
        loss = 2.0 - 2. * torch.mean((out_a_online * out_b_target).sum(dim=-1))

        # symmetrize the above work
        out_b_online = self._compute_embedding(x_b, age_b, self.online_backbone)
        out_b_online = self.online_pred(self.online_proj(out_b_online))
        out_b_online = F.normalize(out_b_online, p=2, dim=-1)
        with torch.no_grad():
            out_a_target = self._compute_embedding(x_a, age_a, self.target_backbone)
            out_a_target = self.target_proj(out_a_target.detach())  # with stop_gradient and no prediction
            out_a_target = F.normalize(out_a_target, p=2, dim=-1)
        loss += 2.0 - 2. * torch.mean((out_b_online * out_a_target).sum(dim=-1))

        return loss
