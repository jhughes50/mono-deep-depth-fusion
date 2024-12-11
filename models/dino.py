"""
    Jason Hughes
    December 2024

    Wrapper Around 4 channel Dinov2
"""

import torch
from dinov2 import DinoVisionTransformer

class DinoBackbone:
    """ A wrapper around DinoV2 """

    def __init__(self) -> None:
        self.model_ = DinoVisionTransformer(img_size=512, in_chan=4)

    @property
    def model(self) -> DinoVisionTransformer:
        return self.model_

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        assert x.size(1) == 4
        embedding = self.model_(x)

        return embedding

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self(x)
