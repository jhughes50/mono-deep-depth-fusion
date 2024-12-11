"""
    Jason Hughes
    December 2024 

    Mono Deep Depth Fusion
"""
import yaml
import torch
import torch.nn as nn
import torch.nn.functionals as F

from depth_head import DepthHead
from dino import DinoBackbone

from torch import Tensor

from torchvision import transforms


class MonoDeepDepthFusionModule(nn.Module):

    def __init__(self, features : int = 256, out_channels : Tuple[int] = (256, 512, 1024, 1024), use_bn : bool : False, max_depth : float = 100.0, layers : Tuple[int] = (2,5,8,11)) -> None:
        super(MonoDeepDepthFusion, self).__init__()

        self.backbone_ = DinoBackbone()
        self.depth_head_ = DepthHead(self.backbone_.embed_dim, out_channels, features=features, use_bn=use_bn)

        self.layers_ = layers

    def forward(self, x : Tensor, max_depth : float = 20.0) -> Tensor:
        """ Forward pass through complete model
            @param : x Tensor -- input image plus sparse depth
        """
        patch_h, patch_w = x.size(2)//16, x.size(3)//16
        
        embeddings = self.backbone_.get_intermediate_layers(x, self.layers_, return_class_token=True)
        
        depth = self.depth_head(embeddings, (patch_h, patch_w))

        return depth

class MonoDeepDepthFusion(MonoDeepDepthFusionModule):

    def __init__(self, features : int = 256, out_channels : Tuple[int] = (256, 512, 1024, 1024), use_bn : bool : False, max_depth : float = 100.0, layers : Tuple[int] = (2,5,8,11)) -> None:
        super(MonoDeepDepthFusionModule, self).__init__(features, out_channels, use_bn, max_depth, layers)

        with open("./config/dino.yaml") as f:
            params = yaml.safe_load(f)

        mean = params["mean"]
        std = params["mean"]

        self.transform_ = transforms.Compose([transforms.Resize(height, width),
                                             transforms.Normalize(mean=mean, std=std)])


    def preprocess(self, rgb : Tensor, sparse_depth : Tensor) -> Tensor:
        rgb = rgb / 255.0
        rgb = self.transform_(rgb)
        return torch.cat([rgb, sparse_depth], dim=1)

    def infer(self, rgb : Tensor, sparse_depth : Tensor, max_depth : float = 20.0) -> Tensor:
        img_with_depth = self.preprocess(rgb, sparse_depth)

        depth = self.forward(img_with_depth)

        depth = depth * max_depth 

        return depth
