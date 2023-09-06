"""
MixUp regularization:
    - PyTorch source code implemented by referring to the link below.
    - https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    - MixUp paper: https://arxiv.org/abs/1710.09412
"""
import numpy as np
import torch


def mixup_data(x, age, y, alpha, device):
    lam = np.random.beta(alpha, alpha) if alpha > 1e-12 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_age = lam * age + (1 - lam) * age[index]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_age, y_a, y_b, lam, index


def mixup_criterion(criterion, pred, y_a, y_b, lam, **kwargs):
    return lam * criterion(pred, y_a, **kwargs) + (1 - lam) * criterion(pred, y_b, **kwargs)
