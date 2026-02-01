import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionMetrics:
    """
    COCO Detection Metrics (mAP, mAP50, mAP75, Recall)
    """

    def __init__(self):
        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            box_format="xyxy"
        )

    def reset(self):
        self.metric.reset()

    def update(self, detections, targets):
        """
        detections: List[Tensor] each (N,6) = [x1,y1,x2,y2,score,label]
        targets:    List[Dict] with keys:
                    boxes (M,4), labels (M,)
        """

        preds = []

        for i, det in enumerate(detections):

            device = targets[i]["boxes"].device

            # No prediction case
            if det is None or len(det) == 0:
                preds.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            preds.append({
                "boxes": det[:, :4],
                "scores": det[:, 4],
                "labels": det[:, 5].long(),
            })

        self.metric.update(preds, targets)

    def compute(self):

        res = self.metric.compute()

        return {
            "mAP": res["map"].item(),
            "mAP50": res["map_50"].item(),
            "mAP75": res["map_75"].item(),
            "Recall@100": res["mar_100"].item(),
        }
