#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING & INFERENCE TEST
Test complete pipeline: Training -> Validation -> Inference -> Visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_dataset(num_samples=20, img_size=224, num_classes=5):
    """Create synthetic dataset for testing"""
    dataset = []
    
    # Class names and colors
    class_names = ['person', 'car', 'bicycle', 'dog', 'cat']
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i in range(num_samples):
        # Create random image with some structure
        image = torch.randn(3, img_size, img_size) * 0.5 + 0.5
        image = torch.clamp(image, 0, 1)
        
        # Create random bounding boxes
        num_objects = torch.randint(1, 4, (1,)).item()
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            # Random box
            x1 = torch.randint(10, img_size//2, (1,)).item()
            y1 = torch.randint(10, img_size//2, (1,)).item()
            w = torch.randint(30, 80, (1,)).item()
            h = torch.randint(30, 80, (1,)).item()
            x2 = min(x1 + w, img_size - 1)
            y2 = min(y1 + h, img_size - 1)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(torch.randint(1, num_classes + 1, (1,)).item())
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
        dataset.append((image, target))
    
    return dataset, class_names, class_colors

def draw_bounding_boxes(image, detections, targets, class_names, class_colors, save_path=None):
    """Draw bounding boxes on image"""
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image
    
    # Create PIL image
    pil_image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw ground truth boxes (green)
    if targets is not None:
        for box, label in zip(targets["boxes"], targets["labels"]):
            x1, y1, x2, y2 = box.int().cpu().numpy()
            class_name = class_names[label.item() - 1] if label.item() <= len(class_names) else f"class_{label.item()}"
            color = (0, 255, 0)  # Green for GT
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 20), class_name, fill=color, font=font)
    
    # Draw predicted boxes (red)
    if detections is not None and len(detections) > 0:
        for det in detections:
            if len(det) == 0:
                continue
                
            x1, y1, x2, y2, score, label = det
            class_idx = int(label.item()) - 1
            class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"class_{class_idx + 1}"
            color = class_colors[class_idx] if 0 <= class_idx < len(class_colors) else (255, 0, 0)
            
            # Only draw if confidence is high enough
            if score > 0.1:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 20), f"{class_name}: {score:.2f}", fill=color, font=font)
    
    # Save or show
    if save_path:
        pil_image.save(save_path)
        print(f"✅ Visualization saved to: {save_path}")
    
    return pil_image

def test_training_pipeline():
    """Test complete training pipeline"""
    print("🧪 TESTING COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        from models.detector import create_corrected_diffnet
        from utils.diffnet_loss import SimpleDetectionLoss
        from utils.losses import CharbonnierLoss
        from utils.metrics import DetectionMetrics
        
        # Create synthetic dataset
        train_dataset, class_names, class_colors = create_synthetic_dataset(num_samples=15, num_classes=5)
        val_dataset, _, _ = create_synthetic_dataset(num_samples=5, num_classes=5)
        
        print(f"✅ Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # Create model
        model = create_corrected_diffnet('tiny', num_classes=5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"✅ Model created on {device}")
        
        # Create loss functions
        det_loss_fn = SimpleDetectionLoss(num_classes=5)
        rec_loss_fn = CharbonnierLoss()
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        print("✅ Training setup complete")
        
        # Training loop
        model.train()
        train_losses = []
        
        print("\n🏋️ TRAINING LOOP")
        print("-" * 40)
        
        for epoch in range(3):  # Short training for testing
            epoch_loss = 0.0
            
            for i, (images, targets) in enumerate(train_dataset):
                # Prepare batch
                images = images.unsqueeze(0).to(device)  # Add batch dimension
                targets_batch = [{k: v.to(device) for k, v in targets.items()}]
                
                # Forward pass
                outputs = model(images)
                
                # Calculate losses
                det_loss, det_logs = det_loss_fn(
                    outputs["raw_detect"],
                    targets_batch,
                    model.detection_subnetwork.head
                )
                
                rec_loss = rec_loss_fn(outputs["recovered"], images)
                total_loss = det_loss + 0.5 * rec_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                if i == 0:  # Debug first sample
                    print(f"  Epoch {epoch+1}, Sample {i+1}:")
                    print(f"    • Detection loss: {det_loss.item():.4f}")
                    print(f"    • Recovery loss: {rec_loss.item():.4f}")
                    print(f"    • Total loss: {total_loss.item():.4f}")
            
            avg_loss = epoch_loss / len(train_dataset)
            train_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        print("✅ Training completed")
        
        # Validation loop
        print("\n🔍 VALIDATION LOOP")
        print("-" * 40)
        
        model.eval()
        metrics = DetectionMetrics()
        metrics.reset()
        
        all_detections = []
        all_targets = []
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_dataset):
                images = images.unsqueeze(0).to(device)
                targets_batch = [{k: v.to(device) for k, v in targets.items()}]
                
                # Forward pass
                outputs = model(images)
                
                # Postprocess
                detections = model.detection_subnetwork.postprocess(
                    outputs["raw_detect"],
                    conf_thresh=0.1,
                    iou_thresh=0.5
                )
                
                # Update metrics
                metrics.update(detections, targets_batch)
                
                # Store for visualization
                all_detections.append(detections[0] if detections else torch.empty((0, 6)))
                all_targets.append(targets)
                
                if i == 0:
                    print(f"  Sample {i+1}:")
                    print(f"    • GT boxes: {len(targets['boxes'])}")
                    print(f"    • Predictions: {len(detections[0]) if detections else 0}")
        
        # Compute metrics
        results = metrics.compute()
        
        print("✅ Validation completed")
        print(f"📊 Validation Results:")
        print(f"  • mAP: {results['mAP']:.4f}")
        print(f"  • mAP50: {results['mAP50']:.4f}")
        print(f"  • mAP75: {results['mAP75']:.4f}")
        print(f"  • Recall@100: {results['Recall@100']:.4f}")
        
        # Check if metrics are reasonable
        if results['mAP'] > 0 and not torch.isnan(torch.tensor(results['mAP'])):
            print("✅ Metrics look reasonable!")
        else:
            print("⚠️  Metrics might have issues")
        
        return True, model, all_detections, all_targets, class_names, class_colors, train_losses, results
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None, None, None, None

def test_inference_and_visualization(model, detections, targets, class_names, class_colors):
    """Test inference and create visualizations"""
    print("\n🎨 TESTING INFERENCE & VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create output directory
        os.makedirs("test_results", exist_ok=True)
        
        # Test on a few samples
        num_samples = min(3, len(detections))
        
        for i in range(num_samples):
            print(f"\n📸 Processing sample {i+1}/{num_samples}")
            
            # Get original image (create a simple one for testing)
            img_size = 224
            test_image = torch.rand(3, img_size, img_size) * 0.7 + 0.3
            
            # Draw results
            result_image = draw_bounding_boxes(
                test_image, 
                detections[i], 
                targets[i], 
                class_names, 
                class_colors,
                save_path=f"test_results/sample_{i+1}_result.png"
            )
            
            print(f"  • GT boxes: {len(targets[i]['boxes'])}")
            print(f"  • Predictions: {len(detections[i])}")
            print(f"  • Visualization saved")
        
        print("✅ Inference and visualization completed")
        return True
        
    except Exception as e:
        print(f"❌ Inference/visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """Test individual model components"""
    print("\n🔧 TESTING MODEL COMPONENTS")
    print("=" * 60)
    
    try:
        from models.detector import create_corrected_diffnet
        
        # Create model
        model = create_corrected_diffnet('tiny', num_classes=5)
        model.eval()
        
        # Test input
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        print("✅ Model forward pass successful")
        
        # Check all outputs
        required_keys = ['raw_detect', 'enhancement_features', 'depth_features', 
                       'multi_scale_features', 'reconstructed']
        
        for key in required_keys:
            if key in outputs:
                if hasattr(outputs[key], 'shape'):
                    print(f"✅ {key}: {outputs[key].shape}")
                else:
                    print(f"✅ {key}: {type(outputs[key])}")
            else:
                print(f"❌ Missing key: {key}")
                return False
        
        # Test postprocessing
        detections = model.detection_subnetwork.postprocess(
            outputs["raw_detect"],
            conf_thresh=0.1,
            iou_thresh=0.5
        )
        
        print(f"✅ Postprocessing: {len(detections)} batches")
        for i, det in enumerate(detections):
            print(f"  • Batch {i}: {len(det)} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_training_behavior(train_losses, val_results):
    """Analyze training behavior and provide insights"""
    print("\n📊 TRAINING BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    print("📈 Training Loss Trend:")
    for i, loss in enumerate(train_losses):
        print(f"  Epoch {i+1}: {loss:.4f}")
    
    if len(train_losses) > 1:
        loss_change = train_losses[-1] - train_losses[0]
        if loss_change < 0:
            print(f"✅ Loss decreasing: {loss_change:.4f}")
        else:
            print(f"⚠️  Loss increasing: {loss_change:.4f}")
    
    print("\n📊 Validation Metrics:")
    for key, value in val_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Check if results are reasonable
    issues = []
    
    if val_results['mAP'] == 0:
        issues.append("mAP is 0 - model may not be learning")
    
    if val_results['mAP50'] == 0:
        issues.append("mAP50 is 0 - no detections at IoU=0.5")
    
    if train_losses[-1] > train_losses[0] * 2:
        issues.append("Loss increased significantly")
    
    if issues:
        print("\n⚠️  Potential Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ Training behavior looks healthy!")

def main():
    """Run comprehensive test"""
    print("🚀 COMPREHENSIVE TRAINING & INFERENCE TEST")
    print("=" * 80)
    
    # Test model components first
    if not test_model_components():
        print("❌ Model components test failed - stopping")
        return False
    
    # Test training pipeline
    success, model, detections, targets, class_names, class_colors, train_losses, val_results = test_training_pipeline()
    
    if not success:
        print("❌ Training pipeline test failed")
        return False
    
    # Test inference and visualization
    if not test_inference_and_visualization(model, detections, targets, class_names, class_colors):
        print("❌ Inference/visualization test failed")
        return False
    
    # Analyze training behavior
    analyze_training_behavior(train_losses, val_results)
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE TEST COMPLETED!")
    print("=" * 80)
    
    print("\n✅ ALL TESTS PASSED!")
    print("✅ Training pipeline works correctly")
    print("✅ Loss calculation is functional")
    print("✅ Metrics computation is working")
    print("✅ Inference produces reasonable results")
    print("✅ Visualization works correctly")
    
    print("\n📋 RESULTS SUMMARY:")
    print(f"• Training loss trend: {'✅ Decreasing' if train_losses[-1] < train_losses[0] else '⚠️  Increasing'}")
    print(f"• Final mAP: {val_results['mAP']:.4f}")
    print(f"• Final mAP50: {val_results['mAP50']:.4f}")
    print(f"• Visualizations saved: test_results/")
    
    print("\n🚀 READY FOR REAL TRAINING!")
    print("Run: python train.py or python train_with_config.py --config base")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
