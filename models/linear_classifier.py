from typing import Tuple

import torch
import torch.nn as nn

# __all__ = []


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, out_dims: int, seq_length: int,
                 use_age: str, dropout: float = 0.3, **kwargs):
        super().__init__()

        self.use_age = use_age
        if use_age not in ['fc', 'conv', 'embedding', 'no']:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) "
                             f"receives one of ['fc', 'conv', 'embedding', 'no'].")
        elif self.use_age == 'conv':
            in_channels += 1
        elif self.use_age == 'embedding':
            self.age_embedding = torch.nn.Parameter((torch.zeros(1, seq_length * in_channels)))
            torch.nn.init.trunc_normal_(self.age_embedding, std=.02)

        self.sequence_length = seq_length
        current_dims = seq_length * in_channels
        if self.use_age in ['fc', 'conv']:
            current_dims = current_dims + 1

        self.output_length = current_dims
        self.linear = nn.Linear(current_dims, out_dims)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N, C, L = x.size()

        x = x.reshape((N, -1))

        if self.use_age in ['conv', 'fc']:
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        elif self.use_age == 'embedding':
            x = x + self.age_embedding * age.reshape(-1, 1)

        x = self.linear(x)
        x = self.dropout(x)

        if target_from_last != 0:
            raise ValueError(f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                             f"an integer equal to or smaller than fc_stages=0.")
        return x

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)
        return x


class LinearClassifier2D(nn.Module):
    def __init__(self, in_channels: int, out_dims: int, seq_len_2d: Tuple[int],
                 use_age: str, dropout: float = 0.3, **kwargs):
        super().__init__()

        self.use_age = use_age
        if use_age not in ['fc', 'conv', 'embedding', 'no']:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) "
                             f"receives one of ['fc', 'conv', 'embedding', 'no'].")
        elif self.use_age == 'conv':
            in_channels += 1
        elif self.use_age == 'embedding':
            self.age_embedding = torch.nn.Parameter((torch.zeros(1, seq_len_2d[0] * seq_len_2d[1] * in_channels)))
            torch.nn.init.trunc_normal_(self.age_embedding, std=.02)

        self.seq_len_2d = seq_len_2d
        current_dims = seq_len_2d[0] * seq_len_2d[1] * in_channels
        if self.use_age in ['fc', 'conv']:
            current_dims = current_dims + 1

        self.output_length = current_dims
        self.linear = nn.Linear(current_dims, out_dims)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N, C, H, W = x.size()

        x = x.reshape((N, -1))

        if self.use_age in ['conv', 'fc']:
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        elif self.use_age == 'embedding':
            x = x + self.age_embedding * age.reshape(-1, 1)

        x = self.linear(x)
        x = self.dropout(x)

        if target_from_last != 0:
            raise ValueError(f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                             f"an integer equal to or smaller than fc_stages=0.")

        return x

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)
        return x
