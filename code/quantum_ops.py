"""Scheme1's fixed feature maps and quantum gate operations."""
import math
import torch
from torch.nn import functional as F

PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

def rotations(axis, angle):
    p = torch.tensor({'X': [[0, 1], [1, 0]], 'Y': [[0, -1j], [1j, 0]], 'Z': [[1, 0], [0, -1]]}[axis], dtype=torch.complex128, device=angle.device)
    return (angle / 2).cos()[..., None, None] * torch.eye(2, dtype=torch.complex128, device=angle.device) - 1j * (angle / 2).sin()[..., None, None] * p

def relation_features(batch):
    t, d, c, b = batch['streams'].double().unbind(-2)
    return torch.cat((t * d, t * c, b * d, b * c, batch['layout'].double(), F.one_hot(batch['role'], 16).double()), -1)

def phase_channels(streams):
    omega = streams.new_tensor([2 * math.pi, 6 * math.pi, 10 * math.pi])
    angles = streams[..., None, :] * omega[:, None]
    return torch.stack((angles.cos(), angles.sin()), -2) / math.sqrt(streams.shape[-1])

def overlaps(streams):
    d = streams.shape[-1]
    p = phase_channels(streams.double())
    return torch.stack([(p[..., a, :, :, :] * p[..., b, :, :, :]).sum(-2).reshape(*streams.shape[:-2], 3, 8, d // 8).sum(-1) * 8 for a, b in PAIRS], -3)

