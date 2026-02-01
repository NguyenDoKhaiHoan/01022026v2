#!/usr/bin/env python3
"""
Training Script with Configuration Support
Usage:
    python train_with_config.py --config base
    python train_with_config.py --config small --epochs 100
    python train_with_config.py --config tiny --batch_size 32
"""
import torch
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import os

from utils.datasets import COCODetectionDataset, detection_collate_fn
from utils.diffnet_loss import SimpleDetectionLoss
from utils.losses import CharbonnierLoss
from utils.config import load_config
from utils.metrics import DetectionMetrics


def create_model_from_config(config):
    """Create model from configuration"""
    from models.detector import create_corrected_diffnet
    
    return create_corrected_diffnet(
        model_size='base',  # Could be from config
        num_classes=config.get('model.num_classes', 80),
        img_size=tuple(config.get('model.input_size', [224, 224]))
    )


def get_data_loaders(config):
    """Create data loaders from configuration"""
    transform = T.Compose([
        T.Resize(tuple(config.get('model.input_size', [224, 224]))),
        T.ToTensor()
    ])
    
    # Dataset paths - you may need to update these
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
    
    batch_size = config.get('training.batch_size', 16)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader


def validate(model, val_loader, device, config):
    """Validation function"""
    model.eval()
    metrics = DetectionMetrics()
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for step, (images, targets) in enumerate(pbar):
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            out = model(images)
            
            # Postprocess detections
            detections = model.detection_subnetwork.postprocess(
                out["raw_detect"],
                conf_thresh=0.01,
                iou_thresh=0.5
            )
            
            if step == 0:
                print(f"GT boxes: {targets[0]['boxes'].shape}")
                print(f"GT labels: {targets[0]['labels'].unique()}")
                print(f"Predictions: {len(detections[0])} detections")
            
            metrics.update(detections, targets)
    
    # Compute metrics
    results = metrics.compute()
    print(f"\nValidation Results:")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"mAP50: {results['mAP50']:.4f}")
    print(f"mAP75: {results['mAP75']:.4f}")
    print(f"Recall@100: {results['Recall@100']:.4f}\n")
    
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Object Detection Model')
    parser.add_argument('--config', type=str, default='base', 
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model configuration')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading {args.config} configuration...")
    config = load_config(args.config)
    config.print_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Setup optimizer and scheduler
    learning_rate = config.get('training.learning_rate', 1e-4)
    weight_decay = config.get('training.weight_decay', 1e-4)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    epochs = config.get('training.epochs', 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Setup loss functions
    num_classes = config.get('model.num_classes', 80)
    det_loss_fn = SimpleDetectionLoss(num_classes=num_classes)
    rec_loss_fn = CharbonnierLoss()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {config.get('training.batch_size')}")
    
    best_map = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (images, targets) in enumerate(pbar):
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            optimizer.zero_grad()
            out = model(images)
            
            # Compute losses
            det_loss, det_logs = det_loss_fn(
                out["raw_detect"],
                targets,
                model.detection_subnetwork.head  # Now head is always available
            )
            
            rec_loss = rec_loss_fn(out["recovered"], images)
            total_loss = det_loss + 0.5 * rec_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "det": f"{det_loss.item():.4f}",
                "rec": f"{rec_loss.item():.4f}"
            })
        
        avg_loss = running_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            results = validate(model, val_loader, device, config)
            
            # Save best model
            current_map = results['mAP']
            if current_map > best_map:
                best_map = current_map
                best_model_path = os.path.join(args.save_dir, f'best_{args.config}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.config,
                    'map': current_map,
                }, best_model_path)
                print(f"New best model saved with mAP: {best_map:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    print(f"Training completed! Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
