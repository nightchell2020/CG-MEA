# Modified from
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
import torch.nn as nn
from typing import Callable, Optional, Type, Union, List, Any

__all__ = ["ResNet2D",
           "resnet18_2d",
           "resnet34_2d",
           "resnet50_2d",
           "resnet101_2d",
           "resnet152_2d",
           "resnext50_32x4d_2d",
           "resnext101_32x8d_2d",
           "wide_resnet50_2d",
           "wide_resnet101_2_2d",
           "BasicBlock2D",
           "Bottleneck2D"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2D(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            current_channels: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock2D only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock2D")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(current_channels, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2D(nn.Module):
    # Bottleneck2D in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            current_channels: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(current_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock2D, Bottleneck2D]],
            conv_layers: List[int],
            fc_stages,
            in_channels,
            out_dims,
            use_age,
            final_pool,
            base_channels,
            n_fft,
            complex_mode,
            hop_length,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs
    ) -> None:
        super(ResNet2D, self).__init__()

        if use_age not in {'fc', 'conv', None}:
            raise ValueError("use_age must be set to one of ['fc', 'conv', None]")

        if final_pool not in {'average', 'max'}:
            raise ValueError("final_pool must be set to one of ['average', 'max']")

        self.use_age = use_age
        self.final_shape = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.zero_init_residual = zero_init_residual
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.current_channels = base_channels
        self.groups = groups
        self.base_width = width_per_group

        if complex_mode not in ('as_real', 'power', 'remove'):
            raise ValueError('complex_mode must be set to one of ("as_real", "power", "remove")')
        self.n_fft = n_fft
        self.complex_mode = complex_mode
        self.hop_length = hop_length

        if complex_mode == 'as_real':
            in_channels *= 2
        in_channels = in_channels + 1 if self.use_age == 'conv' else in_channels

        self.conv1 = nn.Conv2d(in_channels, self.current_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.current_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, conv_layers[0])
        self.layer2 = self._make_layer(block, 128, conv_layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, conv_layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, conv_layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if final_pool == 'average':
            self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif final_pool == 'max':
            self.final_pool = nn.AdaptiveMaxPool2d((1, 1))

        fc_conv_layers = []
        n_current = 512 * block.expansion

        if self.use_age == 'fc':
            n_current = n_current + 1

        for l in range(fc_stages):
            layer = nn.Sequential(nn.Linear(n_current, n_current // 2, bias=False),
                                  nn.Dropout(p=0.1),
                                  nn.BatchNorm1d(n_current // 2),
                                  nn.ReLU())
            n_current = n_current // 2
            fc_conv_layers.append(layer)
        fc_conv_layers.append(nn.Linear(n_current, out_dims))
        self.fc_stage = nn.Sequential(*fc_conv_layers)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def get_final_shape(self):
        return self.final_shape

    def _make_layer(self, block: Type[Union[BasicBlock2D, Bottleneck2D]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.current_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.current_channels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        conv_layers = []
        conv_layers.append(block(self.current_channels, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.current_channels = planes * block.expansion
        for _ in range(1, blocks):
            conv_layers.append(block(self.current_channels, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*conv_layers)

    def _forward_impl(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        N = x.shape[0]

        for i in range(N):
            xf = torch.stft(x[i], n_fft=self.n_fft, return_complex=True)

            if i == 0:
                if self.complex_mode == 'as_real':
                    x_out = torch.zeros((N, 2 * xf.shape[0],
                                         xf.shape[1], xf.shape[2])).type_as(x)
                else:
                    x_out = torch.zeros((N, *xf.shape)).type_as(x)

            if self.complex_mode == 'as_real':
                x_out[i] = torch.cat((torch.view_as_real(xf)[..., 0],
                                      torch.view_as_real(xf)[..., 1]), dim=0)
            elif self.complex_mode == 'power':
                x_out[i] = xf.abs()
            elif self.complex_mode == 'remove':
                x_out[i] = torch.real(xf)
        x = x_out

        if self.use_age == 'conv':
            N, _, H, W = x.size()
            age = age.reshape((N, 1, 1, 1)).expand(N, 1, H, W)
            x = torch.cat((x, age), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.final_shape is None:
            self.final_shape = x.shape

        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        if self.use_age == 'fc':
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)
        x = self.fc_stage(x)

        return x

    def forward(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, age)


def _resnet_2d(
    arch: str,
    block: Type[Union[BasicBlock2D, Bottleneck2D]],
    conv_layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet2D:
    model = ResNet2D(block, conv_layers, **kwargs)
    return model


def resnet18_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d('resnet18_2d', BasicBlock2D, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d('resnet34_2d', BasicBlock2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d('resnet50_2d', Bottleneck2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d('resnet101_2d', Bottleneck2D, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d('resnet152_2d', Bottleneck2D, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_2d('resnext50_32x4d_2d', Bottleneck2D, [3, 4, 6, 3],
                      pretrained, progress, **kwargs)


def resnext101_32x8d_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_2d('resnext101_32x8d_2d', Bottleneck2D, [3, 4, 23, 3],
                      pretrained, progress, **kwargs)


def wide_resnet50_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the Bottleneck2D number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet_2d('wide_resnet50_2d', Bottleneck2D, [3, 4, 6, 3],
                      pretrained, progress, **kwargs)


def wide_resnet101_2_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the Bottleneck2D number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet_2d('wide_resnet101_2_2d', Bottleneck2D, [3, 4, 23, 3],
                      pretrained, progress, **kwargs)