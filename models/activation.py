import torch.nn as nn
import torch.nn.functional as F


activation_list = [
    'relu',
    'gelu',
    'mish',
]


def get_activation_class(activation_type: str, class_name: str = ''):
    if activation_type == 'relu':
        return nn.ReLU
    elif activation_type == 'gelu':
        return nn.GELU
    elif activation_type == 'mish':
        return nn.Mish
    else:
        header = class_name + ', ' if len(class_name) > 0 else ''
        raise ValueError(f"{header}get_activation_class(activation)")


def get_activation_functional(activation_type: str, class_name: str = ''):
    if activation_type == 'relu':
        return F.relu
    elif activation_type == 'gelu':
        return F.gelu
    elif activation_type == 'mish':
        return F.mish
    else:
        header = class_name + ', ' if len(class_name) > 0 else ''
        raise ValueError(f"{header}get_activation_functional(activation)")



