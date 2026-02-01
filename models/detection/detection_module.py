"""
Detection Module - Corrected Version with Multi-scale Feature Pyramid
Implements lightweight detection with multi-scale predictions
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import math

from .common import ConvBNSiLU
from .head import DetectionHead


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    Implements top-down pathway with lateral connections
    """
    
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=256):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.lateral_convs.append(lateral_conv)
        
        # Output convolutions (3x3 conv)
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.output_convs.append(output_conv)
    
    def forward(self, features):
        """
        Args:
            features: List of features from high to low resolution [P3, P4, P5]
            
        Returns:
            pyramid_features: List of FPN features at same resolution
        """
        # Lateral connections
        lateral_features = []
        for i, feature in enumerate(features):
            lateral = self.lateral_convs[i](feature)
            lateral_features.append(lateral)
        
        # Top-down pathway
        pyramid_features = []
        for i in range(len(lateral_features) - 1, -1, -1):
            if i == len(lateral_features) - 1:
                # Highest level, no top-down connection
                pyramid_feature = lateral_features[i]
            else:
                # Upsample higher level and add
                higher_feature = F.interpolate(
                    pyramid_features[-1], 
                    size=lateral_features[i].shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                pyramid_feature = lateral_features[i] + higher_feature
            
            # Apply output conv
            pyramid_feature = self.output_convs[i](pyramid_feature)
            pyramid_features.append(pyramid_feature)
        
        # Reverse to get [P3, P4, P5] order
        pyramid_features.reverse()
        
        return pyramid_features


class MultiScaleDetectionModule(nn.Module):
    """
    ✅ Corrected Detection Module with Multi-scale Feature Pyramid
    
    Key Features:
    - Receives multi-scale features from Transformation Module
    - Uses Feature Pyramid Network for feature fusion
    - Lightweight detection heads for each scale
    - Multi-scale predictions for different object sizes
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        in_channels_list: List[int] = [64, 128, 256],
        fpn_channels: int = 256,
        use_fpn: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        self.fpn_channels = fpn_channels
        self.use_fpn = use_fpn
        
        # Feature Pyramid Network
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=fpn_channels
            )
            detection_in_channels = [fpn_channels] * len(in_channels_list)
        else:
            self.fpn = None
            detection_in_channels = in_channels_list
        
        # Lightweight detection heads for each scale
        self.detection_heads = nn.ModuleList()
        for in_channels in detection_in_channels:
            head = DetectionHead(
                num_classes=num_classes,
                in_channels=[in_channels]  # Single scale per head
            )
            self.detection_heads.append(head)
        
        # Scale-specific anchor generation
        self.strides = [8, 16, 32]  # Standard strides for P3, P4, P5
        
    def _make_anchors(self, features: List[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor points for each scale."""
        anchors = []
        strides = []
        
        for i, feat in enumerate(features):
            _, _, H, W = feat.shape
            stride = self.strides[i]
            
            # Generate grid centers
            sy, sx = torch.meshgrid(
                torch.arange(H, device=device) + 0.5,
                torch.arange(W, device=device) + 0.5,
                indexing='ij'
            )
            anchor = torch.stack([sx, sy], dim=-1).view(-1, 2) * stride
            anchors.append(anchor)
            strides.append(torch.full((H * W,), stride, device=device))
        
        return torch.cat(anchors, dim=0), torch.cat(strides, dim=0)
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            multi_scale_features: List of features from Transformation Module [P3, P4, P5]
            
        Returns:
            outputs: Detection outputs with multi-scale predictions
        """
        # Apply FPN if enabled
        if self.use_fpn:
            fpn_features = self.fpn(multi_scale_features)
        else:
            fpn_features = multi_scale_features
        
        # Generate anchors
        anchors, strides = self._make_anchors(fpn_features, fpn_features[0].device)
        
        # Detection heads for each scale
        cls_outputs = []
        reg_outputs = []
        
        for i, (features, head) in enumerate(zip(fpn_features, self.detection_heads)):
            # Single scale features for this head
            scale_features = [features]
            
            # Forward through detection head
            head_outputs = head(scale_features)
            
            cls_outputs.append(head_outputs["cls_scores"])
            reg_outputs.append(head_outputs["box_regs"])
        
        # Concatenate all scales
        cls_scores = torch.cat(cls_outputs, dim=1)  # (B, total_anchors, C)
        box_regs = torch.cat(reg_outputs, dim=1)    # (B, total_anchors, 4)
        
        return {
            "cls_scores": cls_scores,
            "box_regs": box_regs,
            "anchors": anchors,
            "strides_tensor": strides,
            "fpn_features": fpn_features  # For debugging/visualization
        }
    
    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """Multi-scale post-processing with NMS."""
        from torchvision.ops import nms
        
        cls_scores = outputs['cls_scores'].sigmoid()
        reg_outputs = outputs['box_regs']
        anchors = outputs['anchors']
        strides = outputs['strides_tensor']
        
        # Decode boxes to xyxy
        boxes = self.decode_boxes(reg_outputs, anchors, strides)
        
        B = cls_scores.shape[0]
        results = []
        
        for b in range(B):
            # Get best class per anchor
            scores, class_ids = cls_scores[b].max(dim=-1)
            mask = scores > conf_thres
            
            if mask.sum() == 0:
                results.append(torch.zeros((0, 6), device=boxes.device))
                continue
            
            # Filter by confidence
            filtered_boxes = boxes[b][mask]
            filtered_scores = scores[mask]
            filtered_classes = class_ids[mask]
            
            # NMS
            keep_idx = nms(filtered_boxes, filtered_scores, iou_thres)
            
            # Limit detections
            if len(keep_idx) > max_det:
                keep_idx = keep_idx[:max_det]
            
            final_boxes = filtered_boxes[keep_idx]
            final_scores = filtered_scores[keep_idx]
            final_classes = filtered_classes[keep_idx]
            
            # Pack results
            detections = torch.cat([
                final_boxes,
                final_scores.unsqueeze(-1),
                final_classes.float().unsqueeze(-1)
            ], dim=-1)
            
            results.append(detections)
        
        return results
    
    def decode_boxes(self, reg_outputs, anchors, strides):
        """
        Decode ltrb offsets → xyxy
        
        reg_outputs: (B,N,4)
        """
        B = reg_outputs.shape[0]
        reg_outputs = torch.relu(reg_outputs)  # Ensure positive offsets
        
        lt = reg_outputs[..., :2]
        rb = reg_outputs[..., 2:]
        
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)
        strides = strides.unsqueeze(0).unsqueeze(-1).expand(B, -1, 2)
        
        x1y1 = anchors - lt * strides
        x2y2 = anchors + rb * strides
        
        return torch.cat([x1y1, x2y2], dim=-1)


class LightweightDetectionSubnetwork(nn.Module):
    """
    Complete Detection Subnetwork with Multi-scale capabilities
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        in_channels_list: List[int] = [64, 128, 256],
        fpn_channels: int = 256,
        use_fpn: bool = True
    ):
        super().__init__()
        
        self.detection_module = MultiScaleDetectionModule(
            num_classes=num_classes,
            in_channels_list=in_channels_list,
            fpn_channels=fpn_channels,
            use_fpn=use_fpn
        )
        
        # For compatibility with existing code - always provide first head
        self.head = self.detection_module.detection_heads[0]
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.detection_module(features)
    
    def postprocess(self, outputs: Dict[str, torch.Tensor], 
                   conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[torch.Tensor]:
        return self.detection_module.postprocess(outputs, conf_thres, iou_thres)
