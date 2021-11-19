import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = []


class TinyCNN1D(nn.Module):
    def __init__(self, in_channels, out_dims, use_age, final_pool,
                 stride=7, base_channels=64, **kwargs):
        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

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

        if self.use_age == 'fc':
            self.fc1 = nn.Linear(base_channels + 1, base_channels)
        else:
            self.fc1 = nn.Linear(base_channels, base_channels)

        self.dropout = nn.Dropout(p=0.3)
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

        # conv-bn-relu-pool
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

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


class M7(nn.Module):
    def __init__(self, in_channels, out_dims, use_age, final_pool,
                 base_channels=256, **kwargs):
        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

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

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        if self.use_age == 'fc':
            self.fc1 = nn.Linear(2 * base_channels + 1, 2 * base_channels)
        else:
            self.fc1 = nn.Linear(2 * base_channels, 2 * base_channels)

        self.dropout = nn.Dropout(p=0.3)
        self.bnfc1 = nn.BatchNorm1d(2 * base_channels)
        self.fc2 = nn.Linear(2 * base_channels, out_dims)

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

        # conv-bn-relu-pool
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)

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
