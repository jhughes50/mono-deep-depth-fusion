"""
    Jason Hughes
    Devember 2024

    A depth head for depth completion
"""

import torch
import torch.nn as nn

from utils.feature_handler import DinoFeatureHandler
from torch import Tensor

class DepthHead(nn.Module):

    def __init__(self, in_channels : int) -> None:
        super(DepthHead, self).__init__()



    def forward(self, embeddings : Tuple[Tensor]) -> torch.Tensor:
        
        pass


