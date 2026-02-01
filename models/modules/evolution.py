"""
Transformation Module - Corrected Version with Dual Inputs
Receives both original image and fused features from Feature Mining branch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module for combining Enhancement and Depth features
    Implements: f_ED = Concat(f_E, f_D)
    """
    
    def __init__(self, enh_dim=64, depth_dim=64, fusion_dim=128):
        super().__init__()
        
        self.enh_dim = enh_dim
        self.depth_dim = depth_dim
        self.fusion_dim = fusion_dim
        
        # Projection layers to ensure same dimensions
        self.enh_proj = nn.Sequential(
            nn.Conv2d(enh_dim, fusion_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_dim, fusion_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Fusion attention mechanism
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim // 8, 2, kernel_size=1),  # 2 for enh and depth
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, enh_features, depth_features):
        """
        Args:
            enh_features: Features from Enhancement Module [B, enh_dim, H, W]
            depth_features: Features from Depth Module [B, depth_dim, H, W]
            
        Returns:
            fused_features: Combined features [B, fusion_dim, H, W]
        """
        # Project to fusion dimension
        enh_proj = self.enh_proj(enh_features)      # [B, fusion_dim//2, H, W]
        depth_proj = self.depth_proj(depth_features) # [B, fusion_dim//2, H, W]
        
        # Simple concatenation
        concat_features = torch.cat([enh_proj, depth_proj], dim=1)  # [B, fusion_dim, H, W]
        
        # Attention-based fusion
        attention_weights = self.fusion_attention(concat_features)  # [B, 2, H, W]
        enh_weight = attention_weights[:, 0:1, :, :]  # [B, 1, H, W]
        depth_weight = attention_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Weighted fusion
        weighted_enh = enh_proj * enh_weight
        weighted_depth = depth_proj * depth_weight
        weighted_fusion = torch.cat([weighted_enh, weighted_depth], dim=1)
        
        # Output projection
        fused_features = self.output_proj(weighted_fusion)
        
        return fused_features


class DualInputTransformationModule(nn.Module):
    """
    ✅ Corrected Transformation Module with Dual Inputs
    
    Key Changes:
    - Receives both original image AND fused features
    - Combines spatial details (image) with semantic features (fused)
    - Outputs multi-scale features for detection
    """
    
    def __init__(self, image_channels=3, feature_channels=128, 
                 encoding_channels=[64, 128, 256]):
        super().__init__()
        
        self.image_channels = image_channels
        self.feature_channels = feature_channels
        self.encoding_channels = encoding_channels
        
        # Input processing for image branch (spatial details)
        self.image_processor = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Input processing for feature branch (semantic guidance)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion of processed inputs
        self.input_fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 64+64=128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale encoding layers
        self.encoders = nn.ModuleList()
        
        in_channels = 128
        for out_channels in encoding_channels:
            encoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.encoders.append(encoder_block)
            in_channels = out_channels
        
        # Multi-scale feature outputs
        self.multi_scale_outputs = encoding_channels
        
    def forward(self, image, fused_features):
        """
        Args:
            image: Original input image [B, 3, H, W]
            fused_features: Features from Feature Fusion [B, feature_channels, H, W]
            
        Returns:
            multi_scale_features: List of features at different scales
        """
        # Process image branch (spatial details)
        image_features = self.image_processor(image)  # [B, 64, H/4, W/4]
        
        # Process feature branch (semantic guidance)
        processed_features = self.feature_processor(fused_features)  # [B, 64, H, W]
        
        # Ensure same spatial resolution
        if processed_features.shape[2:] != image_features.shape[2:]:
            processed_features = F.interpolate(
                processed_features, 
                size=image_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Fuse the two branches
        combined_input = torch.cat([image_features, processed_features], dim=1)  # [B, 128, H/4, W/4]
        fused_input = self.input_fusion(combined_input)  # [B, 128, H/4, W/4]
        
        # Multi-scale encoding
        multi_scale_features = []
        current_features = fused_input
        
        for encoder in self.encoders:
            current_features = encoder(current_features)
            multi_scale_features.append(current_features)
        
        return multi_scale_features


class CorrectedRecoveryModule(nn.Module):
    """
    Recovery Module (Decoder) for reconstruction
    """
    
    def __init__(self, encoded_channels=256, decoding_channels=[128, 64, 32], out_channels=3):
        super().__init__()
        
        self.decoders = nn.ModuleList()
        
        in_channels = encoded_channels
        for out_channels in decoding_channels:
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.decoders.append(decoder_block)
            in_channels = out_channels
        
        # Final output layer
        self.output_conv = nn.Conv2d(decoding_channels[-1], out_channels, kernel_size=3, padding=1)
        
    def forward(self, encoded_features):
        """
        Args:
            encoded_features: Features from Transformation Module [B, C, H, W]
            
        Returns:
            reconstructed: Reconstructed output [B, out_channels, H, W]
            decoding_features: List of decoding features
        """
        decoding_features = []
        current_features = encoded_features
        
        for decoder in self.decoders:
            current_features = decoder(current_features)
            decoding_features.append(current_features)
        
        reconstructed = self.output_conv(current_features)
        
        return reconstructed, decoding_features


class CorrectedFeatureEvolutionSubnetwork(nn.Module):
    """
    ✅ Corrected Feature Evolution Subnetwork
    
    Implements the complete Feature Evolution pipeline:
    1. Feature Fusion (f_E + f_D → f_ED)
    2. Dual-input Transformation (Image + f_ED → Multi-scale features)
    3. Recovery (Decoder)
    """
    
    def __init__(self, image_channels=3, enh_dim=64, depth_dim=64,
                 fusion_dim=128, encoding_channels=[64, 128, 256],
                 decoding_channels=[128, 64, 32], out_channels=3):
        super().__init__()
        
        # 1. Feature Fusion Module
        self.feature_fusion = FeatureFusionModule(
            enh_dim=enh_dim,
            depth_dim=depth_dim,
            fusion_dim=fusion_dim
        )
        
        # 2. Dual-input Transformation Module
        self.transformation = DualInputTransformationModule(
            image_channels=image_channels,
            feature_channels=fusion_dim,
            encoding_channels=encoding_channels
        )
        
        # 3. Recovery Module
        self.recovery = CorrectedRecoveryModule(
            encoded_channels=encoding_channels[-1],
            decoding_channels=decoding_channels,
            out_channels=out_channels
        )
        
    def forward(self, image, enh_features, depth_features):
        """
        Args:
            image: Original input image [B, 3, H, W]
            enh_features: Features from Enhancement Module [B, enh_dim, H, W]
            depth_features: Features from Depth Module [B, depth_dim, H, W]
            
        Returns:
            reconstructed: Reconstructed output [B, out_channels, H, W]
            multi_scale_features: Features from Transformation Module
            decoding_features: Features from Recovery Module
        """
        # Step 1: Feature Fusion
        fused_features = self.feature_fusion(enh_features, depth_features)
        
        # Step 2: Dual-input Transformation
        multi_scale_features = self.transformation(image, fused_features)
        
        # Step 3: Recovery
        reconstructed, decoding_features = self.recovery(multi_scale_features[-1])
        
        return reconstructed, multi_scale_features, decoding_features
