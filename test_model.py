#!/usr/bin/env python3
"""
Test script to check if model produces reasonable outputs
"""
import torch
from models.detector import DiffNet

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = DiffNet(num_classes=5).to(device)
    model.eval()
    
    # Test input
    batch_size = 2
    img_size = 448
    test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        # Forward pass
        outputs = model(test_input)
        
        print("\n=== Model Outputs ===")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape} | min: {value.min().item():.4f} | max: {value.max().item():.4f}")
            elif isinstance(value, list):
                print(f"{key}: list with {len(value)} items")
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    print(f"  First item: {value[0].shape}")
        
        # Check detection outputs specifically
        if "raw_detect" in outputs:
            raw_det = outputs["raw_detect"]
            print(f"\n=== Detection Details ===")
            print(f"Cls scores shape: {raw_det['cls_scores'].shape}")
            print(f"Box regs shape: {raw_det['box_regs'].shape}")
            print(f"Anchors shape: {raw_det['anchors'].shape}")
            print(f"Strides shape: {raw_det['strides_tensor'].shape}")
            
            # Check if model produces reasonable confidence scores
            cls_probs = torch.sigmoid(raw_det['cls_scores'])
            max_conf = cls_probs.max().item()
            mean_conf = cls_probs.mean().item()
            print(f"Max confidence: {max_conf:.6f}")
            print(f"Mean confidence: {mean_conf:.6f}")
            
            # Count predictions above threshold
            high_conf_mask = cls_probs > 0.1
            num_high_conf = high_conf_mask.sum().item()
            print(f"Number of predictions > 0.1: {num_high_conf}")
            
            if num_high_conf == 0:
                print("⚠️  WARNING: No predictions above 0.1 confidence!")
                print("This might indicate initialization issues.")
            else:
                print("✅ Model produces reasonable confidence scores")

if __name__ == "__main__":
    test_model()
