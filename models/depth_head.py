"""
    Jason Hughes
    Devember 2024

    A depth head for depth completion
"""

import torch
import torch.nn as nn

from utils.feature_handler import DinoFeatureHandler
from utils.feature_fusion import FeatureFusionModule 
from utils.layers import make_scratch
from torch import Tensor

class DepthHead(nn.Module):

    def __init__(self, in_channels : int, out_channels : Tuple[int], features : int = 256, use_bn : bool = False) -> None:
        super(DepthHead, self).__init__()

        self.patch_h_, self.patch_w_ = patch

        self.feature_handler_ = DinoFeatureHandler(in_channels, out_channels)

        self.scratch_ = make_scratch(out_channels, features)

        self.scratch_.stem_transpose = None

        self.scratch_.refinenet1 = FeatureFusionModule(featues, "relu", use_bn=use_bn)
        self.scratch_.refinenet2 = FeatureFusionModule(featues, "relu", use_bn=use_bn)
        self.scratch_.refinenet3 = FeatureFusionModule(featues, "relu", use_bn=use_bn)
        self.scratch_.refinenet4 = FeatureFusionModule(featues, "relu", use_bn=use_bn)

        self.scratch_.output_conv1 = nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1)
        self.scratch_.output_conv2 = nn.Sequential(nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
                                                   nn.ReLU(True),
                                                   nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
                                                   nn.Sigmoid())

    def forward(self, embeddings : Tuple[Tensor], patch_size : Tuple[int]) -> Tensor:
        """ forward pass for the depth head 
            @param: embedding Tuple[Tuple[Tensor]] -- output embeddings from dino layers

            @returns: tbd
        """
        # handle the embeddings, get a list of 4 back and unpack them
        layer1, layer2, layer3, layer4  = self.feature_handler_(embeddings, patch_size[0], patch_size[1])

        layer1_rn = self.scratch_.layer1_rn(layer1)
        layer2_rn = self.scratch_.layer2_rn(layer2)
        layer3_rn = self.scratch_.layer3_rn(layer3)
        layer4_rn = self.scratch_.layer4_rn(layer4)

        path_4 = self.scratch_.refinenet4(layer4_rn, size=layer3.shape[2:])
        path_3 = self.scratch_.refinenet3(path_4, layer3_rn, size=layer2_rn.shape[2:])
        path_2 = self.scratch_.refinenet2(path_3, layer2_rn, size=layer1_rn.shape[2:])
        path_1 = self.scratch_.refinenet1(path_2, layer1_rn)

        out = self.scratch_output_conv1(path_1)
        out = F.interpolate(out, (int(self.patch_h_ * 14), int(self.patch_w_ * 14)), mode= "bilinear", align_corners=True)
        out = self.scratch_.output_conv2(out)

        return out 
