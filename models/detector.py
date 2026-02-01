"""
Corrected DiffNet - Implementation of the Complete 3-Branch Architecture
Implements the exact flow described in the requirements
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

# Import corrected modules (now renamed to main files)
from .modules.enhancement import FeatureEnhancementBlock
from .modules.depth import FeatureDepthModuleWithProjection
from .modules.evolution import CorrectedFeatureEvolutionSubnetwork
from .detection.detection_module import LightweightDetectionSubnetwork


class CorrectedDiffNet(nn.Module):
    """
    ✅ Corrected DiffNet - Complete 3-Branch Architecture
    
    Architecture Flow:
    
    🌟 LUỒNG 1 - Feature Mining (Encoder):
    Input Image → Enhancement Module (ViT) → f_E (features)
    f_E → Depth Module (Swin) → f_D (features)
    f_E + f_D → Feature Fusion → f_ED (fused features)
    
    🌟 LUỒNG 2 - Feature Evolution (Song song):
    Input Image + f_ED → Transformation Module → multi-scale features
    features → Recovery Module → reconstructed image
    
    🌟 LUỒNG 3 - Detection:
    multi-scale features → Detection Module → Bounding boxes + Classes
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        img_size: Tuple[int, int] = (224, 224),
        enhancement_cfg: Dict = None,
        depth_cfg: Dict = None,
        evolution_cfg: Dict = None,
        detection_cfg: Dict = None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Default configurations
        if enhancement_cfg is None:
            enhancement_cfg = {
                'embed_dim': 64,
                'num_heads': 4,
                'num_blocks': 2,
                'window_size': 7,
                'patch_size': 4,
                'cardinality': 16
            }
        
        if depth_cfg is None:
            depth_cfg = {
                'in_feature_dim': enhancement_cfg['embed_dim'],
                'embed_dim': 96,
                'out_feature_dim': enhancement_cfg['embed_dim'],
                'depth': 3,
                'num_heads': 8,
                'window_size': 7
            }
        
        if evolution_cfg is None:
            evolution_cfg = {
                'image_channels': 3,
                'enh_dim': enhancement_cfg['embed_dim'],
                'depth_dim': depth_cfg['out_feature_dim'],
                'fusion_dim': enhancement_cfg['embed_dim'] * 2,
                'encoding_channels': [64, 128, 256],
                'decoding_channels': [128, 64, 32],
                'out_channels': 3
            }
        
        if detection_cfg is None:
            detection_cfg = {
                'num_classes': num_classes,
                'in_channels_list': evolution_cfg['encoding_channels'],
                'fpn_channels': 256,
                'use_fpn': True
            }
        
        # ==============================================
        # LUỒNG 1 - FEATURE MINING (ENCODER)
        # ==============================================
        
        # 1.1 Enhancement Module (ViT-based)
        # Input: Image [B, 3, H, W]
        # Output: f_E features [B, embed_dim, H/patch_size, W/patch_size]
        self.enhancement_module = FeatureEnhancementBlock(
            in_channels=3,
            **enhancement_cfg
        )
        
        # 1.2 Depth Module (Swin-based)
        # Input: f_E features from Enhancement
        # Output: f_D features [B, embed_dim, H/patch_size, W/patch_size]
        self.depth_module = FeatureDepthModuleWithProjection(
            **depth_cfg
        )
        
        # ==============================================
        # LUỒNG 2 - FEATURE EVOLUTION (SONG SONG)
        # ==============================================
        
        # 2.1 Complete Feature Evolution Subnetwork
        # Includes: Feature Fusion + Dual-input Transformation + Recovery
        self.feature_evolution = CorrectedFeatureEvolutionSubnetwork(
            **evolution_cfg
        )
        
        # ==============================================
        # LUỒNG 3 - DETECTION
        # ==============================================
        
        # 3.1 Lightweight Detection Subnetwork
        # Input: Multi-scale features from Transformation
        # Output: Bounding boxes + Classes
        self.detection_subnetwork = LightweightDetectionSubnetwork(
            **detection_cfg
        )
        
        print(f"✅ CorrectedDiffNet initialized with {num_classes} classes")
        print(f"📊 Enhancement: {enhancement_cfg['embed_dim']}dim, {enhancement_cfg['num_blocks']} blocks")
        print(f"🌊 Depth: {depth_cfg['depth']} blocks, {depth_cfg['embed_dim']}dim")
        print(f"🔄 Evolution: {evolution_cfg['encoding_channels']} channels")
        print(f"🎯 Detection: Multi-scale with FPN")
    
    def forward(
        self, 
        x: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing the complete 3-branch architecture
        
        Args:
            x: Input image tensor [B, 3, H, W]
            conf_thres: Confidence threshold for inference
            iou_thres: IoU threshold for NMS
            
        Returns:
            Dictionary containing all intermediate and final outputs
        """
        B, C, H, W = x.shape
        
        # ==============================================
        # LUỒNG 1 - FEATURE MINING (ENCODER)
        # ==============================================
        
        # Step 1.1: Enhancement Module (ViT)
        # Input: Image [B, 3, H, W]
        # Output: f_E features [B, embed_dim, H/patch_size, W/patch_size]
        f_E = self.enhancement_module(x)
        
        # Step 1.2: Depth Module (Swin)
        # Input: f_E features from Enhancement
        # Output: f_D features [B, embed_dim, H/patch_size, W/patch_size]
        f_D = self.depth_module(f_E)
        
        # ==============================================
        # LUỒNG 2 - FEATURE EVOLUTION (SONG SONG)
        # ==============================================
        
        # Step 2.1: Complete Feature Evolution
        # Input: Original Image + f_E + f_D
        # Output: reconstructed + multi_scale_features + decoding_features
        reconstructed, multi_scale_features, decoding_features = self.feature_evolution(
            image=x,
            enh_features=f_E,
            depth_features=f_D
        )
        
        # ==============================================
        # LUỒNG 3 - DETECTION
        # ==============================================
        
        # Step 3.1: Detection Module
        # Input: multi_scale_features from Transformation
        # Output: Detection predictions
        raw_outputs = self.detection_subnetwork(multi_scale_features)
        
        # Post-processing for inference
        if not self.training:
            detections = self.detection_subnetwork.postprocess(
                raw_outputs, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres
            )
        else:
            detections = None
        
        # ==============================================
        # RETURN ALL OUTPUTS FOR DEBUGGING AND MONITORING
        # ==============================================
        
        return {
            # Final outputs
            'detections': detections,
            'raw_detect': raw_outputs,
            'reconstructed': reconstructed,
            
            # Feature Mining branch outputs
            'enhancement_features': f_E,
            'depth_features': f_D,
            
            # Feature Evolution branch outputs
            'multi_scale_features': multi_scale_features,
            'decoding_features': decoding_features,
            
            # Original inputs for reference
            'input_image': x
        }
    
    def get_model_info(self):
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'CorrectedDiffNet',
            'num_classes': self.num_classes,
            'input_size': self.img_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'feature_mining': {
                    'enhancement': type(self.enhancement_module).__name__,
                    'depth': type(self.depth_module).__name__
                },
                'feature_evolution': {
                    'evolution': type(self.feature_evolution).__name__
                },
                'detection': {
                    'detection': type(self.detection_subnetwork).__name__
                }
            }
        }
        
        return info
    
    def print_model_summary(self):
        """Print model summary"""
        info = self.get_model_info()
        
        print("=" * 60)
        print("🚀 CORRECTED DIFFNET MODEL SUMMARY")
        print("=" * 60)
        print(f"📋 Model Name: {info['model_name']}")
        print(f"🎯 Classes: {info['num_classes']}")
        print(f"📐 Input Size: {info['input_size']}")
        print(f"🔢 Total Parameters: {info['total_parameters']:,}")
        print(f"🔧 Trainable Parameters: {info['trainable_parameters']:,}")
        print("")
        print("🏗️  ARCHITECTURE:")
        print("  🌟 LUỒNG 1 - Feature Mining:")
        print(f"    • Enhancement: {info['architecture']['feature_mining']['enhancement']}")
        print(f"    • Depth: {info['architecture']['feature_mining']['depth']}")
        print("  🌟 LUỒNG 2 - Feature Evolution:")
        print(f"    • Evolution: {info['architecture']['feature_evolution']['evolution']}")
        print("  🌟 LUỒNG 3 - Detection:")
        print(f"    • Detection: {info['architecture']['detection']['detection']}")
        print("=" * 60)


# Factory function for easy model creation
def create_corrected_diffnet(
    model_size: str = 'base',
    num_classes: int = 7,
    img_size: Tuple[int, int] = (224, 224)
) -> CorrectedDiffNet:
    """
    Factory function to create CorrectedDiffNet with predefined configurations
    
    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        num_classes: Number of detection classes
        img_size: Input image size
        
    Returns:
        CorrectedDiffNet instance
    """
    
    # Configuration presets
    configs = {
        'tiny': {
            'enhancement_cfg': {
                'embed_dim': 32,
                'num_heads': 4,
                'num_blocks': 2,
                'window_size': 7,
                'patch_size': 4,
                'cardinality': 16
            },
            'depth_cfg': {
                'in_feature_dim': 32,
                'embed_dim': 48,
                'out_feature_dim': 32,
                'depth': 2,
                'num_heads': 4,
                'window_size': 7
            },
            'evolution_cfg': {
                'image_channels': 3,
                'enh_dim': 32,
                'depth_dim': 32,
                'fusion_dim': 64,
                'encoding_channels': [32, 64, 128],
                'decoding_channels': [64, 32, 16],
                'out_channels': 3
            }
        },
        'small': {
            'enhancement_cfg': {
                'embed_dim': 48,
                'num_heads': 6,
                'num_blocks': 3,
                'window_size': 7,
                'patch_size': 4,
                'cardinality': 24
            },
            'depth_cfg': {
                'in_feature_dim': 48,
                'embed_dim': 64,
                'out_feature_dim': 48,
                'depth': 3,
                'num_heads': 6,
                'window_size': 7
            },
            'evolution_cfg': {
                'image_channels': 3,
                'enh_dim': 48,
                'depth_dim': 48,
                'fusion_dim': 96,
                'encoding_channels': [48, 96, 192],
                'decoding_channels': [96, 48, 24],
                'out_channels': 3
            }
        },
        'base': {
            'enhancement_cfg': {
                'embed_dim': 64,
                'num_heads': 8,
                'num_blocks': 3,
                'window_size': 7,
                'patch_size': 4,
                'cardinality': 32
            },
            'depth_cfg': {
                'in_feature_dim': 64,
                'embed_dim': 96,
                'out_feature_dim': 64,
                'depth': 3,
                'num_heads': 8,
                'window_size': 7
            },
            'evolution_cfg': {
                'image_channels': 3,
                'enh_dim': 64,
                'depth_dim': 64,
                'fusion_dim': 128,
                'encoding_channels': [64, 128, 256],
                'decoding_channels': [128, 64, 32],
                'out_channels': 3
            }
        },
        'large': {
            'enhancement_cfg': {
                'embed_dim': 96,
                'num_heads': 12,
                'num_blocks': 4,
                'window_size': 7,
                'patch_size': 4,
                'cardinality': 32
            },
            'depth_cfg': {
                'in_feature_dim': 96,
                'embed_dim': 128,
                'out_feature_dim': 96,
                'depth': 4,
                'num_heads': 12,
                'window_size': 7
            },
            'evolution_cfg': {
                'image_channels': 3,
                'enh_dim': 96,
                'depth_dim': 96,
                'fusion_dim': 192,
                'encoding_channels': [96, 192, 384],
                'decoding_channels': [192, 96, 48],
                'out_channels': 3
            }
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size '{model_size}' not supported. Choose from: {list(configs.keys())}")
    
    config = configs[model_size]
    
    model = CorrectedDiffNet(
        num_classes=num_classes,
        img_size=img_size,
        **config
    )
    
    return model


# Alias for backward compatibility
DiffNet = CorrectedDiffNet
