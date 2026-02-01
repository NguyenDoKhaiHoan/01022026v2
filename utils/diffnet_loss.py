# utils/diffnet_loss.py

import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss


# ==================================================
# Assign GT boxes to nearest anchors (improved assign)
# ==================================================
def assign_targets_to_anchors(gt_boxes, anchors, strides):
    """
    Assign each GT box to nearest anchor (center distance + IoU)

    Args:
        gt_boxes: (M,4) xyxy
        anchors:  (N,2)
        strides:  (N,)

    Returns:
        assigned_idx: (M,)
        pos_mask: (N,) - positive anchors
        neg_mask: (N,) - negative anchors
    """

    # GT centers (M,2)
    gt_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2.0

    # Distance matrix (M,N)
    dist = torch.cdist(gt_centers, anchors)
    
    # Each GT chooses closest anchor
    assigned_idx = dist.argmin(dim=1)
    
    # Create positive/negative masks
    pos_mask = torch.zeros(len(anchors), dtype=torch.bool, device=gt_boxes.device)
    pos_mask[assigned_idx] = True
    
    # Ignore anchors that are too far from any GT
    min_dist, _ = dist.min(dim=0)
    neg_mask = min_dist > 3.0  # threshold in pixels
    
    return assigned_idx, pos_mask, neg_mask


# ==================================================
# Simple Detection Loss
# ==================================================
class SimpleDetectionLoss(nn.Module):
    """
    Loss for Simple Detection Head:

    Outputs:
        cls_scores: (B,N,C)
        box_regs:   (B,N,4)

    Loss:
        - Classification: BCEWithLogitsLoss
        - Box regression: GIoU Loss (positives only)

    Total = cls_loss + box_weight * box_loss
    """

    def __init__(self, num_classes=7, box_weight=2.0):
        super().__init__()

        self.num_classes = num_classes
        self.box_weight = box_weight

        self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, det_outputs, targets, head):

        pred_cls = det_outputs["cls_scores"]   # (B,N,C)
        pred_reg = det_outputs["box_regs"]    # (B,N,4)

        anchors = det_outputs["anchors"]      # (N,2)
        strides = det_outputs["strides_tensor"]

        B, N, C = pred_cls.shape

        # Decode predicted boxes → xyxy
        pred_boxes = head.decode_boxes(pred_reg, anchors, strides)

        # ==================================================
        # IMPORTANT: Loss must be Tensor (not float)
        # ==================================================
        total_cls_loss = torch.tensor(0.0, device=pred_cls.device)
        total_box_loss = torch.tensor(0.0, device=pred_cls.device)

        # ==================================================
        # Loop over batch
        # ==================================================
        for b in range(B):

            gt_boxes = targets[b]["boxes"].to(pred_cls.device)
            gt_labels = targets[b]["labels"].to(pred_cls.device)

            M = gt_boxes.shape[0]

            # Case: no GT objects
            if M == 0:
                neg_target = torch.zeros_like(pred_cls[b])
                total_cls_loss += self.cls_loss_fn(pred_cls[b], neg_target)
                continue

            # Assign GT → anchors with improved assignment
            assigned_idx, pos_mask, neg_mask = assign_targets_to_anchors(
                gt_boxes, anchors, strides
            )

            # ==================================================
            # 1. Classification Loss
            # ==================================================
            cls_target_full = torch.zeros_like(pred_cls[b])  # (N,C)

            # Positive samples: assign GT class
            for j in range(M):
                a = assigned_idx[j]
                cls_id = int(gt_labels[j])
                # Ensure class_id is within valid range [0, num_classes-1]
                cls_id = max(0, min(cls_id - 1, self.num_classes - 1))  # Convert to 0-based
                cls_target_full[a, cls_id] = 1.0

            # Apply focal-like weighting for positive/negative balance
            pos_weight = torch.where(pos_mask.float().unsqueeze(-1), 2.0, 1.0)
            neg_weight = torch.where(neg_mask.float().unsqueeze(-1), 1.0, 0.1)
            sample_weight = pos_weight * neg_weight

            loss_cls = (self.cls_loss_fn(pred_cls[b], cls_target_full) * sample_weight).mean()

            # ==================================================
            # 2. Box Regression Loss (GIoU) - only for positives
            # ==================================================
            if pos_mask.sum() > 0:
                pred_pos_boxes = pred_boxes[b][pos_mask]
                gt_pos_boxes = gt_boxes  # Use all GT boxes for simplicity
                
                # Ensure we have matching number of boxes
                if len(pred_pos_boxes) == len(gt_pos_boxes):
                    loss_box = generalized_box_iou_loss(
                        pred_pos_boxes,
                        gt_pos_boxes,
                        reduction="mean"
                    )
                else:
                    # Fallback: use only assigned boxes
                    assigned_pos_boxes = pred_boxes[b, assigned_idx]
                    loss_box = generalized_box_iou_loss(
                        assigned_pos_boxes,
                        gt_boxes,
                        reduction="mean"
                    )
            else:
                loss_box = torch.tensor(0.0, device=pred_cls.device)

            total_cls_loss += loss_cls
            total_box_loss += loss_box

        # ==================================================
        # Normalize over batch
        # ==================================================
        total_cls_loss = total_cls_loss / B
        total_box_loss = total_box_loss / B

        total_loss = total_cls_loss + self.box_weight * total_box_loss

        # ==================================================
        # Return loss + logs
        # ==================================================
        return total_loss, {
            "loss_cls": total_cls_loss.item(),
            "loss_box": total_box_loss.item(),
        }
