"""
    Jason Hughes
    Decemeber 2024

    Feature Fusion Block...
    Taken from depth anything v2 with some changes
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

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

        self.residual_conv1_ = ResidualConvModule(features, activation, bn)
        self.residual_conv2_ = ResidualConvModule(features, activation, bn)

        self.skip_add_ = nn.quantized.FloatFunctional()

        self.size_ = size


    def forward(self, x : Tensor, size : int = None) -> Tensor:
        output = x[0]

        if len(x) == 2:
            residual = self.residual_conv1_(x[1])
            output = self.skip_add_.add(output, residual)

        output = self.residual_conv2_(output)

        if (size is None) and (self.size_ is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size_}
        else:
            modifier = {"size": size}

        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners_)
        output = self.out_conv(output)

        return output


class ResidualConvModule(nn.Module):

    def __init__(self, features : int, activation : str, bn : bool):
        super(ResidualConvModule, self).__init__()

        self.bn_ = bn
        self.groups_ = 1

        self.conv1_ = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups_)
        self.conv2_ = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups_)

        if self.bn_:
            self.bn1_ = nn.BatchNorm2d(features)
            self.bn2_ = nn.BatchNorm2d(features)

        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        elif activation == "softmax":
            activation = nn.Softmax()
        else:
            print("[RESIDUALMODULE] Activation not supported, defaulting to ReLU")
            activation = nn.ReLU()
        self.activation_ = activation

        self.skip_add_ = nn.quantized.FloatFunctional()

    def forward(self, x : Tensor) -> Tensor:
        
        out = self.conv1_(x)
        if self.bn_:
            out = self.bn1_(out)
        out = self.activation_(out)

        out = self.conv2_(out)
        if self.bn_:
            out = self.bn2_(out)
        out = self.activation_(out)

        if self.groups_ > 1:
            out = self.conv_merge(out)

        output = self.skip_add_.add(out, x)

        return output
