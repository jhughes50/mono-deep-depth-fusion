"""
    Jason Hughes
    December 2024

    Dataset object that loads images
"""

import torch

from torch.utils.data import Dataset

class UnsupervisedDataset(Dataset):

    def __init__(self, path : str = "./") -> None:
        super(UnsupervisedDataset, self).__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, iter : int) -> torch.Tensor:
        # get an image
        pass
