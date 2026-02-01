from torch.utils.data import DataLoader
from dataset import COCODetectionDataset, detection_collate_fn
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

train_dataset = COCODetectionDataset(
    img_dir="/kaggle/input/dtaaaaaaaa/dataset/train/img",
    ann_file="/kaggle/input/dtaaaaaaaa/dataset/train/anno/_annotations.coco.json",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=detection_collate_fn,
    num_workers = 0
)
