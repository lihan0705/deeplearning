"""
vision transformer for image from scratch

better version: $ pip install vit-pytorch

"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    split image into patches and do embedding

    parameters:
        img_size : int
        patch_size : int
        in_channels: int
        embed_dim : int
    Attributes:
        n_patches : int
        proj : nn.Conv2d

    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patch = (img_size // patch_size)**2  # must be nxn images?

        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=self.patch_size)

    def forward(self, x):
        pass
