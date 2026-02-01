import torch
from torchvision.ops import nms


def postprocess_detections(raw_detect, head,
                           conf_thresh=0.01,
                           iou_thresh=0.5):
    """
    Convert raw head outputs → COCO detections format

    Returns:
        List[Tensor] each (N,6)
        [x1,y1,x2,y2,score,label]
    """

    cls_scores = raw_detect["cls_scores"]   # (B,N,C)
    box_regs   = raw_detect["box_regs"]     # (B,N,4)

    anchors = raw_detect["anchors"]
    strides = raw_detect["strides_tensor"]

    # Decode boxes → xyxy
    boxes = head.decode_boxes(box_regs, anchors, strides)

    B = cls_scores.shape[0]
    detections_batch = []

    for b in range(B):

        # Sigmoid logits → prob
        probs = torch.sigmoid(cls_scores[b])   # (N,C)

        # Best class per anchor
        scores, labels = probs.max(dim=1)

        # Filter confidence
        keep = scores > conf_thresh

        final_boxes  = boxes[b][keep]
        final_scores = scores[keep]
        final_labels = labels[keep]

        if len(final_boxes) == 0:
            detections_batch.append(
                torch.zeros((0, 6), device=boxes.device)
            )
            continue

        # NMS
        keep_idx = nms(final_boxes, final_scores, iou_thresh)

        final_boxes  = final_boxes[keep_idx]
        final_scores = final_scores[keep_idx]
        final_labels = final_labels[keep_idx]

        # Pack output (N,6)
        det = torch.cat([
            final_boxes,
            final_scores.unsqueeze(1),
            final_labels.unsqueeze(1).float()
        ], dim=1)

        detections_batch.append(det)

    return detections_batch
