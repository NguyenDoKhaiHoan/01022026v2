import torch
import torch.nn as nn
from typing import List, Dict
import math

from .common import ConvBNSiLU


class DWConv(nn.Module):
    """Depthwise Separable Conv"""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.dw = ConvBNSiLU(in_ch, in_ch, k, groups=in_ch)
        self.pw = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class DetectionHead(nn.Module):
    """
    Simple Detection Head (NO DFL, NO reg_max)

    Outputs:
        cls_scores: (B,N,C)
        box_regs:   (B,N,4)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: List[int] = [256, 512, 1024],
    ):
        super().__init__()

        self.num_classes = num_classes

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for c in in_channels:

            # Classification branch
            self.cls_heads.append(nn.Sequential(
                DWConv(c, c),
                DWConv(c, c),
                nn.Conv2d(c, num_classes, 1)
            ))

            # Regression branch (4 only!)
            self.reg_heads.append(nn.Sequential(
                ConvBNSiLU(c, c, 3),
                ConvBNSiLU(c, c, 3),
                nn.Conv2d(c, 4, 1)
            ))

        self._init_bias()

    def _init_bias(self):
        """Stable cls bias init - less negative"""
        for head in self.cls_heads:
            conv = head[-1]
            b = conv.bias.view(-1)
            # Giảm từ -math.log((1 - 0.01) / 0.01) ≈ -4.6 xuống -2.3
            b.data.fill_(-math.log((1 - 0.1) / 0.1))  # 0.1 instead of 0.01

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        cls_outputs = []
        reg_outputs = []

        for i, feat in enumerate(features):

            B, C, H, W = feat.shape

            cls_map = self.cls_heads[i](feat)
            reg_map = self.reg_heads[i](feat)

            # Flatten
            cls_out = cls_map.view(B, self.num_classes, -1).permute(0, 2, 1)
            reg_out = reg_map.view(B, 4, -1).permute(0, 2, 1)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)

        return {
            "cls_scores": torch.cat(cls_outputs, dim=1),  # (B,N,C)
            "box_regs": torch.cat(reg_outputs, dim=1),    # (B,N,4)
        }

    def decode_boxes(self, reg_outputs, anchors, strides):
        """
        Decode ltrb offsets → xyxy

        reg_outputs: (B,N,4)
        """

        B = reg_outputs.shape[0]
        reg_outputs = torch.relu(reg_outputs)

        lt = reg_outputs[..., :2]
        rb = reg_outputs[..., 2:]

        anchors = anchors.unsqueeze(0).expand(B, -1, -1)
        strides = strides.unsqueeze(0).unsqueeze(-1).expand(B, -1, 2)

        x1y1 = anchors - lt * strides
        x2y2 = anchors + rb * strides

        return torch.cat([x1y1, x2y2], dim=-1)
