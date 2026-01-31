"""
Enhancement Module - FIXED VERSION
Low-light enhancement with Swin Transformer + ResNeXt

✅ Fix OOM by using Patch Embedding (Downsample before Swin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .common import (
    SwinTransformerBlock,
    init_weights,
    create_stochastic_depth_decay
)

# ============================================================
# ResNeXt Bottleneck Form C (Grouped Conv)
# ============================================================

class ResNeXtBottleneckC(nn.Module):
    def __init__(self, in_channels, out_channels,
                 cardinality=16, base_width=4, stride=1):
        super().__init__()

        D = int(math.floor(out_channels * (base_width / 64)) * cardinality)

        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)

        self.conv2 = nn.Conv2d(
            D, D, kernel_size=3,
            stride=stride, padding=1,
            groups=cardinality,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(D)

        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        return self.relu(out)


# ============================================================
# Patch Embedding (Downsample before Swin)
# ============================================================

class PatchEmbed(nn.Module):
    """
    Convert image -> patch tokens

    Input:  [B, 3, 224, 224]
    Output: [B, embed_dim, 56, 56]  (patch_size=4)
    """

    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# ============================================================
# Low Light Enhancement Block (FIXED)
# ============================================================

class LowLightEnhancementBlock(nn.Module):
    """
    ✅ Fixed Enhancement Module

    Key Fix:
    - Downsample input before Swin
    - Swin works on 56×56 tokens not 224×224
    - Upsample back to original resolution
    """

    def __init__(
        self,
        in_channels=3,
        embed_dim=64,
        num_heads=4,
        window_size=7,
        num_blocks=2,
        patch_size=4,
        mlp_ratio=4.0,
        cardinality=16,
        base_width=4,
        drop_path_rate=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # ✅ Patch Embedding (Downsample)
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # Stochastic depth decay
        dpr = create_stochastic_depth_decay(num_blocks, drop_path_rate)

        # Swin + ResNeXt blocks
        self.swin_blocks = nn.ModuleList()
        self.resnext_blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.swin_blocks.append(
                SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=(56, 56),  # will be updated dynamically
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm
                )
            )

            self.resnext_blocks.append(
                ResNeXtBottleneckC(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    cardinality=cardinality,
                    base_width=base_width
                )
            )

        # Output projection
        self.output_conv = nn.Conv2d(
            embed_dim, in_channels,
            kernel_size=3,
            padding=1
        )

        init_weights(self)

    def forward(self, x):
        """
        Input:  [B, 3, H, W]
        Output: [B, 3, H, W]
        """

        identity = x
        B, C, H, W = x.shape

        # ====================================================
        # ✅ Patch Embedding Downsample
        # ====================================================
        x = self.patch_embed(x)
        Hs, Ws = x.shape[2], x.shape[3]

        # ====================================================
        # Swin + ResNeXt enhancement
        # ====================================================
        for i in range(self.num_blocks):

            # --- Swin path ---
            x_seq = x.flatten(2).transpose(1, 2)

            self.swin_blocks[i].input_resolution = (Hs, Ws)

            x_seq = self.swin_blocks[i](x_seq)

            x_swin = x_seq.transpose(1, 2).view(B, self.embed_dim, Hs, Ws)

            # --- ResNeXt path ---
            x_res = self.resnext_blocks[i](x)

            # Fusion
            x = x_swin + x_res

        # ====================================================
        # ✅ Upsample back to original resolution
        # ====================================================
        x = F.interpolate(
            x,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # Output conv
        x = self.output_conv(x)

        # Residual add
        out = x + identity
        return out


# ============================================================
# Factory Function
# ============================================================

def create_enhancement_model():
    """
    Safe default config for Kaggle GPU
    """

    return LowLightEnhancementBlock(
        embed_dim=64,
        num_heads=4,
        window_size=7,
        num_blocks=2,
        patch_size=4,
        cardinality=16
    )
