"""
    Jason Hughes
    December 2024

    handler the residual features from dinov2
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple, Union

class FeatureProjectionModule(nn.Module):

    def __init__(self, in_channels : int, out_channels : Tuple[int]) -> None:
        super(FeatureProjectionModule, self).__init__()
        self.projection_2d_ = nn.ModuleList([nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channel, 
                                                      kernel_size=1, 
                                                      stride=1, 
                                                      padding=0) for out_channel in out_channels])
        self.resize_layers_ = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0],
                                                                out_channels=out_channels[0],
                                                                kernel_size=4,
                                                                stride=4,
                                                                padding=0),
                                             nn.ConvTranspose2d(in_channels=out_channels[1],
                                                                out_channels=out_channels[1],
                                                                kernel_size=2,
                                                                stride=2,
                                                                padding=0),
                                             nn.Identity(),
                                             nn.Conv2d(in_channels=out_channels[3],
                                                       out_channels=out_channels[3],
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1)])

    def forward(self, x : Tensor, iter : int) -> Tensor:
        x = self.projection_2d_[iter](x)
        x = self.resize_layers_(x)


class DinoFeatureHandler(FeatureProjectionModule):

    def __init__(self, patch_height : int, patch_width : int, in_channels : int, out_channels : Tuple[int] ) -> None:
        super().__init__(in_channels, out_channels)

        self.patch_height_ = patch_height
        self.patch_width_  = patch_width

    def __call__(self, features : Tuple) -> List[Tensor]:
        output = list()
        for iter, x in enumerate(features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.size(0), x.shape[-1], self.patch_height_, self.patch_weight_))
            
            x = forward(x, iter)
            
            output.append(x)

        return output


if __name__ == "__main__":
    """ debug call """
    dfh = DinoFeatureHandler(16, 16, 4, (256,256,256,256))
