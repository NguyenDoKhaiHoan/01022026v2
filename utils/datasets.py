# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class COCODetectionDataset(Dataset):
    """
    Custom COCO-style Dataset for DiffNet Detection

    Returns:
        image: Tensor [3,H,W]
        target: dict {
            "boxes": Tensor[N,4] (xyxy)
            "labels": Tensor[N]
        }
    """

    def __init__(self, img_dir, ann_file, transform=None):
        """
        Args:
            img_dir: path to images folder
            ann_file: path to COCO json annotation file
            transform: torchvision transforms
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load COCO annotation json
        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Build mapping: image_id -> annotations
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # ---- Image info ----
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_dir, file_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # ---- Load annotations ----
        anns = self.img_id_to_anns.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            # COCO bbox format = [x, y, w, h]
            x, y, w, h = ann["bbox"]

            # Convert to xyxy
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])

            # category_id -> label
            labels.append(ann["category_id"]+1)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        # ---- Transform ----
        if self.transform is not None:
            image = self.transform(image)

        return image, target


# =====================================================
# Collate Function (REQUIRED for detection)
# =====================================================

def detection_collate_fn(batch):
    """
    Detection batch collate:
    Returns:
        images: list[Tensor]
        targets: list[dict]
    """
    return tuple(zip(*batch))
