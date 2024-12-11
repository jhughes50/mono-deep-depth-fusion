"""
    Jason Hughes
    December 2024

    Make the scratch layers -- FROM DAv2
"""
import torch
import torch.nn as nn

from torch import Tensor

from typeing import Tuple

def make_scratch(in_shape : Tuple[int], out_shape : int, groups : int = 1, expand : bool = False) -> nn.Module:
    scratch = nn.Module()

    s0, s1, s2, s3 = out_shape, out_shape, out_shape, out_shape

    if exapand:
        s1 = s1 * 2
        s2 = s2 * 4
        s3 = s3 * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], s0, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], s1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], s2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], s3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch
