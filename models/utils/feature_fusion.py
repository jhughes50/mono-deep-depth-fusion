"""
    Jason Hughes
    Decemeber 2024

    Feature Fusion Block...
    Taken from depth anything v2 with some changes
"""

import torch 
import torch.nn as nn

from torch import Tensor

class FeatureFusionModule(nn.Module):

    def __init__(self, features : int, activation : str, deconv : bool = False, bn : bool = False, expand : bool = False, align_corner : bool = False, size : Tuple[int] = None) -> None:
        super(FeatureFusionModule, self).__init__()

        self.deconv_ = deconv
        self.align_corners_ = align_corners
        self.groups_ = 1

        self.expand_ = expand

        out_features = features
        if self.expand_:
            out_features = features // 2

        self.out_conv_ = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        if activation == "relu":
            activation == nn.ReLU()
        elif activation == "sigmoid":
            activation == nn.Sigmoid()
        elif activation == "softmax":
            activation == nn.Softmax()
        else:
            print("[FEATURE-FUSION-MODULE] Activation not supported, defaulting to ReLU")
            activation = nn.ReLU()

    def forward(self, x : Tensor) -> Tensor:
        pass


class ResidualConvModule(nn.Module):

    def __init__(self, features : int, activation : str, bn : bool):
        super(ResidualConvModule, self).__init__()

    def forward(self, x : Tensor) -> Tensor:
        pass
