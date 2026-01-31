import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from utils.datasets import COCODetectionDataset, detection_collate_fn
from utils.diffnet_loss import SimpleDetectionLoss
from utils.losses import CharbonnierLoss
from utils.postprocess import postprocess_detections
from models.detector import DiffNet   # ví dụ model bạn

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Transform
# -----------------------------
transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor()
])

# -----------------------------
# 3. Dataset + DataLoader
# -----------------------------
train_dataset = COCODetectionDataset(
    img_dir="/kaggle/input/dtaaaaaaaa/dataset/train/img",
    ann_file="/kaggle/input/dtaaaaaaaa/dataset/train/anno/_annotations.coco.json",
    transform=transform
)
val_dataset = COCODetectionDataset(
    img_dir="/kaggle/input/dtaaaaaaaa/dataset/valid/img",
    ann_file="/kaggle/input/dtaaaaaaaa/dataset/valid/anno/_annotations.coco.json",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=detection_collate_fn,
    num_workers = 0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=detection_collate_fn,
    num_workers=0
)

# -----------------------------
# 4. Model
# -----------------------------
model = DiffNet().to(device)

# Better optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=5,  # số epochs
    eta_min=1e-6
)

# -----------------------------
# 5. Loss functions
# -----------------------------
det_loss_fn = SimpleDetectionLoss(num_classes=5)
rec_loss_fn = CharbonnierLoss()

# -----------------------------
# 6. Training loop
# -----------------------------
from tqdm import tqdm

print("Start training loop...")
from utils.metrics import DetectionMetrics

metrics = DetectionMetrics()


def validate(model, val_loader, device):

    model.eval()
    metrics.reset()

    with torch.no_grad():

        pbar = tqdm(val_loader, desc="Validating")

        for step, (images, targets) in enumerate(pbar):

            # Move images
            images = torch.stack(images).to(device)

            # Move targets
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            # Forward
            out = model(images)

            # Postprocess → detections
            detections = postprocess_detections(
                out["raw_detect"],
                model.detection_subnetwork.head,
                conf_thresh=0.01,  # Giảm từ 0.3 xuống 0.01
                iou_thresh=0.5
            )

            if step == 0:
                print("GT boxes:", targets[0]["boxes"].shape)
                print("GT labels:", targets[0]["labels"].unique())
                
                # Debug model outputs
                cls_scores = out["raw_detect"]["cls_scores"]
                print("Raw cls scores shape:", cls_scores.shape)
                print("Raw cls scores min/max:", cls_scores.min().item(), cls_scores.max().item())
                
                print("Pred det:", detections[0])
                print("Num det:", len(detections[0]))

            # Update metric
            metrics.update(detections, targets)

    # Compute final COCO metrics
    results = metrics.compute()

    print("\n COCO Validation Results")
    print(f"mAP        : {results['mAP']:.4f}")
    print(f"mAP50      : {results['mAP50']:.4f}")
    print(f"mAP75      : {results['mAP75']:.4f}")
    print(f"Recall@100 : {results['Recall@100']:.4f}\n")

    model.train()


for epoch in range(5):

    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")

    for step, (images, targets) in enumerate(pbar):

        images = torch.stack(images).to(device)

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        # Forward
        out = model(images)

        # Detection loss
        det_loss, det_logs = det_loss_fn(
            out["raw_detect"],
            targets,
            model.detection_subnetwork.head
        )

        # Recovery loss
        rec_loss = rec_loss_fn(out["recovered"], images)

        total_loss = det_loss + 0.5 * rec_loss

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        pbar.set_postfix({
            "total": f"{total_loss.item():.4f}",
            "det": f"{det_loss.item():.4f}",
            "rec": f"{rec_loss.item():.4f}"
        })

    avg_loss = running_loss / len(train_loader)

    print(f"\n Epoch {epoch+1} finished | Avg Loss = {avg_loss:.4f}\n")
    
    # Update learning rate
    scheduler.step()
    print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    validate(model, val_loader,device)


