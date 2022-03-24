import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = []


class TinyCNN1D(nn.Module):
    def __init__(self, in_channels, out_dims, fc_stages, use_age, final_pool,
                 stride=7, base_channels=64, dropout=0.3, activation='relu', **kwargs):
        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

        if activation == 'relu':
            self.F_act = F.relu
            self.nn_act = nn.ReLU
        elif activation == 'gelu':
            self.F_act = F.gelu
            self.nn_act = nn.GELU
        elif activation == 'mish':
            self.F_act = F.mish
            self.nn_act = nn.Mish
        else:
            raise ValueError("final_pool must be set to one of ['relu', 'gelu', 'mish']")

        in_channels = in_channels + 1 if self.use_age == 'conv' else in_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=35, stride=stride)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=7)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.pool2 = nn.MaxPool1d(2)

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == 'fc':
            base_channels = base_channels + 1

        for l in range(fc_stages):
            layer = nn.Sequential(nn.Linear(base_channels, base_channels // 2, bias=False),
                                  nn.Dropout(p=dropout),
                                  nn.BatchNorm1d(base_channels // 2),
                                  self.nn_act())
            base_channels = base_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(base_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

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

        # conv-bn-act-pool
        x = self.conv1(x)
        x = self.F_act(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.F_act(self.bn2(x))
        x = self.pool2(x)

        if self.final_shape is None:
            self.final_shape = x.shape
        x = self.final_pool(x).reshape((N, -1))

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        x = self.fc_stage(x)

        # return F.log_softmax(x, dim=1)
        return x


class M7(nn.Module):
    def __init__(self, in_channels, out_dims, fc_stages, use_age, final_pool,
                 base_channels=256, dropout=0.3, activation='relu', **kwargs):
        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

        if activation == 'relu':
            self.F_act = F.relu
            self.nn_act = nn.ReLU
        elif activation == 'gelu':
            self.F_act = F.gelu
            self.nn_act = nn.GELU
        elif activation == 'mish':
            self.F_act = F.mish
            self.nn_act = nn.Mish
        else:
            raise ValueError("final_pool must be set to one of ['relu', 'gelu', 'mish']")

        in_channels = in_channels + 1 if self.use_age == 'conv' else in_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=41, stride=2)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=11)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(base_channels, 2 * base_channels, kernel_size=11)
        self.bn3 = nn.BatchNorm1d(2 * base_channels)
        self.pool3 = nn.MaxPool1d(3)

        self.conv4 = nn.Conv1d(2 * base_channels, 2 * base_channels, kernel_size=11)
        self.bn4 = nn.BatchNorm1d(2 * base_channels)
        self.pool4 = nn.MaxPool1d(3)

        self.conv5 = nn.Conv1d(2 * base_channels, 2 * base_channels, kernel_size=11)
        self.bn5 = nn.BatchNorm1d(2 * base_channels)
        self.pool5 = nn.MaxPool1d(2)
        base_channels = 2 * base_channels

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == 'fc':
            base_channels = base_channels + 1

        for l in range(fc_stages):
            layer = nn.Sequential(nn.Linear(base_channels, base_channels // 2, bias=False),
                                  nn.Dropout(p=dropout),
                                  nn.BatchNorm1d(base_channels // 2),
                                  self.nn_act())
            base_channels = base_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(base_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

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

        # conv-bn-act-pool
        x = self.conv1(x)
        x = self.F_act(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.F_act(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.F_act(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.F_act(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.F_act(self.bn5(x))
        x = self.pool5(x)

        if self.final_shape is None:
            self.final_shape = x.shape
        x = self.final_pool(x).reshape((N, -1))

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        x = self.fc_stage(x)

        # return F.log_softmax(x, dim=1)
        return x
