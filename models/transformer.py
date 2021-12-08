import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = []


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(self, in_channels, out_dims, use_age, final_pool,
                 base_channels=256, n_encoders=4, n_heads=2, dropout=0.2, **kwargs):
        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

        in_channels = in_channels + 1 if self.use_age == 'conv' else in_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=21, stride=11)
        self.bn1 = nn.BatchNorm1d(base_channels)

        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=9, stride=3)
        self.bn2 = nn.BatchNorm1d(base_channels)

        self.pos_encoder = PositionalEncoding(base_channels, dropout)
        encoder_layers = nn.TransformerEncoderLayer(base_channels, nhead=n_heads,
                                                    dim_feedforward=base_channels * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoders)

        self.conv3 = nn.Conv1d(base_channels, 2 * base_channels, kernel_size=9, stride=3)
        self.bn3 = nn.BatchNorm1d(2 * base_channels)
        base_channels = 2 * base_channels

        self.conv4 = nn.Conv1d(base_channels, base_channels, kernel_size=9, stride=3)
        self.bn4 = nn.BatchNorm1d(base_channels)

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        if self.use_age == 'fc':
            self.fc1 = nn.Linear(base_channels + 1, base_channels)
        else:
            self.fc1 = nn.Linear(base_channels, base_channels)

        self.dropout = nn.Dropout(p=dropout)
        self.bnfc1 = nn.BatchNorm1d(base_channels)
        self.fc2 = nn.Linear(base_channels, out_dims)

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def get_final_shape(self):
        return self.final_shape

    def forward(self, x, age):
        N, C, L = x.size()

        if self.use_age == 'conv':
            age = age.reshape((N, 1, 1))
            age = torch.cat([age for i in range(L)], dim=2)
            x = torch.cat((x, age), dim=1)

        # conv-bn-relu
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        # conv-bn-relu
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        # transformer encoder layers
        x = x.permute(2, 0, 1)  # minibatch, dimension, length --> length, minibatch, dimension
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # length, minibatch, dimension --> minibatch, dimension, length

        # conv-bn-relu again
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        # conv-bn-relu again
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        if self.final_shape is None:
            self.final_shape = x.shape
        x = self.final_pool(x).reshape((N, -1))

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        # fc-bn-dropout-relu-fc
        x = self.fc1(x)
        x = self.bnfc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)

        # return F.log_softmax(x, dim=1)
        return x