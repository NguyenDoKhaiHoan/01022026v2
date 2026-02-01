"""
Depth Module - Corrected Version for Feature Mining
Receives features from Enhancement Module instead of image
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from .common import Mlp, window_partition, window_reverse, WindowAttention, SwinTransformerBlock, init_weights, create_stochastic_depth_decay


class FeatureDepthModule(nn.Module):
    """
    ✅ Corrected Depth Module
    
    Key Changes:
    - Receives FEATURES from Enhancement Module
    - Outputs FEATURES for downstream processing
    - Maintains spatial resolution from enhancement features
    """
    
    def __init__(self, feature_dim=64, depth=3, num_heads=8, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.feature_dim = feature_dim
        self.depth = depth
        
        # Build blocks with alternating regular and shifted window attention
        self.blocks = nn.ModuleList()
        
        # Stochastic depth decay rule
        dpr = create_stochastic_depth_decay(depth, drop_path_rate)
        
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=feature_dim,
                input_resolution=(56, 56),  # Will be updated dynamically
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            self.blocks.append(block)
        
        # Final normalization
        self.norm = norm_layer(feature_dim)
        
        # Feature projection for better integration
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features from Enhancement Module [B, feature_dim, H, W]
        Returns:
            Output features [B, feature_dim, H, W]
        """
        B, C, H, W = x.shape
        input_resolution = (H, W)
        
        # Convert to sequence format for Swin blocks
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Pass through Swin blocks sequentially
        for i, blk in enumerate(self.blocks):
            # Update input resolution for each block
            blk.input_resolution = input_resolution
            x_seq = blk(x_seq)
        
        # Final normalization
        x_seq = self.norm(x_seq)
        
        # Convert back to image format
        x = x_seq.transpose(1, 2).view(B, C, H, W)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class FeatureDepthModuleWithProjection(nn.Module):
    """
    Enhanced Depth Module with flexible input/output dimensions
    """
    
    def __init__(self, in_feature_dim=64, embed_dim=96, out_feature_dim=64,
                 depth=3, num_heads=8, window_size=7,
                 mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        
        self.in_feature_dim = in_feature_dim
        self.embed_dim = embed_dim
        self.out_feature_dim = out_feature_dim
        
        # Input projection to embedding dimension
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_feature_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Depth module (Swin blocks)
        self.depth_module = FeatureDepthModule(
            feature_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        
        # Output projection to desired feature dimension
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_feature_dim),
            nn.GELU()
        )
        
        # Residual connection if dimensions differ
        if in_feature_dim != out_feature_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_feature_dim, out_feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_feature_dim)
            )
        else:
            self.shortcut = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        init_weights(self)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features from Enhancement Module [B, in_feature_dim, H, W]
        Returns:
            Output features [B, out_feature_dim, H, W]
        """
        identity = self.shortcut(x)
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (B, embed_dim, H, W)
        
        # Process through depth module
        x = self.depth_module(x)  # (B, embed_dim, H, W)
        
        # Project back to output feature dimension
        x = self.output_proj(x)  # (B, out_feature_dim, H, W)
        
        # Residual connection
        x = x + identity
        
        return x


def create_feature_depth_model():
    """
    Factory function for creating depth model
    """
    return FeatureDepthModuleWithProjection(
        in_feature_dim=64,
        embed_dim=96,
        out_feature_dim=64,
        depth=3,
        num_heads=8,
        window_size=7
    )
