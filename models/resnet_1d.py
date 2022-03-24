from typing import Type, Union, List

import torch
import torch.nn as nn

# __all__ = []


class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(self, c_in, c_out, kernel_size, stride, groups=1, activation=nn.ReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=c_out,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)

        self.conv2 = nn.Conv1d(in_channels=c_out, out_channels=c_out, groups=groups,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)

        self.activation = activation()

        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=c_in, out_channels=c_out,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.activation(x + identity)

        return x


class BottleneckBlock1D(nn.Module):
    expansion: int = 4

    def __init__(self, c_in, c_out, kernel_size, stride, groups=1, activation=nn.ReLU) -> None:
        super().__init__()
        width = c_out
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

        self.conv3 = nn.Conv1d(in_channels=width, out_channels=c_out * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c_out * self.expansion)

        self.activation = activation()

        self.downsample = None
        if stride != 1 or c_in != c_out * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=c_in, out_channels=c_out * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.activation(x + identity)

        return x


class MultiBottleneckBlock1D(nn.Module):
    expansion: int = 4

    def __init__(self, c_in, c_out, kernel_size, stride, groups=1, activation=nn.ReLU) -> None:
        super().__init__()
        width = c_out

        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=width, kernel_size=1,
                               dilation=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        self.conv2_1 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                                 kernel_size=kernel_size, dilation=1,
                                 stride=stride, padding=kernel_size // 2, bias=False)
        self.conv2_2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                                 kernel_size=kernel_size, dilation=2,
                                 stride=stride, padding=(kernel_size // 2) * 2, bias=False)
        self.bn2 = nn.BatchNorm1d(width * 2)

        self.conv3 = nn.Conv1d(in_channels=width * 2, out_channels=c_out * self.expansion, kernel_size=1,
                               dilation=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c_out * self.expansion)

        self.activation = activation()

        self.downsample = None
        if stride != 1 or c_in != c_out * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=c_in, out_channels=c_out * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.activation(x + identity)
        return x


class ResNet1D(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock1D, BottleneckBlock1D, MultiBottleneckBlock1D]],
                 conv_layers: List[int],
                 fc_stages: int,
                 in_channels: int,
                 out_dims: int,
                 use_age,
                 final_pool,
                 base_channels=64,
                 first_stride=2,
                 first_dilation=1,
                 base_stride=3,
                 dropout=0.1,
                 groups=1,
                 kernel_size=9,
                 activation='relu',
                 **kwargs) -> None:

        super().__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

        if activation == 'relu':
            self.nn_act = nn.ReLU
        elif activation == 'gelu':
            self.nn_act = nn.GELU
        elif activation == 'mish':
            self.nn_act = nn.Mish
        else:
            raise ValueError("activation must be set to one of ['relu', 'gelu', 'mish']")

        self.groups = groups
        self.current_channels = base_channels
        in_channels = in_channels + 1 if self.use_age == 'conv' else in_channels

        self.input_stage = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=base_channels,
                      kernel_size=kernel_size * 3, stride=first_stride, dilation=first_dilation,
                      padding=(kernel_size * 3) // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            self.nn_act(),
        )

        self.conv_stage1 = self._make_conv_layer(block, conv_layers[0], base_channels,
                                                 kernel_size, stride=base_stride, activation=self.nn_act)
        self.conv_stage2 = self._make_conv_layer(block, conv_layers[1], base_channels * 2,
                                                 kernel_size, stride=base_stride, activation=self.nn_act)
        self.conv_stage3 = self._make_conv_layer(block, conv_layers[2], base_channels * 4,
                                                 kernel_size, stride=base_stride, activation=self.nn_act)
        self.conv_stage4 = self._make_conv_layer(block, conv_layers[3], base_channels * 8,
                                                 kernel_size, stride=base_stride, activation=self.nn_act)

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == 'fc':
            self.current_channels = self.current_channels + 1

        for l in range(fc_stages):
            layer = nn.Sequential(nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                                  nn.Dropout(p=dropout),
                                  nn.BatchNorm1d(self.current_channels // 2),
                                  self.nn_act())
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(self.current_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def get_final_shape(self):
        return self.final_shape

    def _make_conv_layer(self, block: Type[Union[BasicBlock1D, BottleneckBlock1D]], n_block: int,
                         c_out: int, kernel_size: int, stride: int = 1, activation=nn.ReLU) -> nn.Sequential:
        layers = []
        c_in = self.current_channels
        layers.append(block(c_in, c_out, kernel_size, groups=self.groups, stride=1, activation=activation))

        c_in = c_out * block.expansion
        self.current_channels = c_in
        for _ in range(1, n_block):
            layers.append(block(c_in, c_out, kernel_size, groups=self.groups, stride=1, activation=activation))

        layers.append(nn.MaxPool1d(kernel_size=stride))

        return nn.Sequential(*layers)

    def forward(self, x, age):
        N, _, L = x.size()
        if self.use_age == 'conv':
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            x = torch.cat((x, age), dim=1)

        x = self.input_stage(x)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)

        if self.final_shape is None:
            self.final_shape = x.shape
        x = self.final_pool(x).reshape((N, -1))

        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        x = self.fc_stage(x)

        # return F.log_softmax(x, dim=2)
        return x
