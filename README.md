# Object Detection Model

A comprehensive object detection model that combines low-light enhancement, depth processing, and feature evolution for robust object detection in challenging conditions.

## Architecture

The model consists of four main components:

1. **Low-light Enhancement Module** (3 ViT blocks)
   - Combines Swin Transformer and ResNeXt blocks
   - Enhances image quality in low-light conditions

2. **Depth Module** (3 Swin blocks)
   - Processes features through Swin Transformer blocks
   - Captures hierarchical depth information with self-attention

3. **Feature Evolution Subnetwork**
   - **Transformation Module**: Encoder with 3 convolutional layers
   - **Recovery Module**: Decoder with 3 deconvolutional layers

4. **Detection Head**
   - Multi-scale detection for bounding box regression and classification

## Project Structure

```
Object_detection/
├── configs/                 # YAML configuration files
│   ├── model_tiny.yaml     # Tiny model configuration
│   ├── model_small.yaml    # Small model configuration
│   ├── model_base.yaml     # Base model configuration
│   └── model_large.yaml    # Large model configuration
├── models/                 # Model modules
│   ├── __init__.py
│   ├── detector.py         # Main detection model
│   └── modules/           # Individual model components
│       ├── __init__.py
│       ├── depth.py       # Depth Module (Swin blocks)
│       ├── enhancement.py # Enhancement Module (ViT + ResNeXt)
│       ├── evolution.py   # Feature Evolution Subnetwork
│       └── head.py        # Detection Head
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── config.py         # Configuration loader
│   ├── datasets.py       # Dataset utilities
│   └── losses.py         # Loss functions
├── data/                 # Data directories
│   ├── raw/             # Raw dataset files
│   └── processed/       # Processed dataset files
├── train.py             # Training script
├── test.py              # Testing script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Object_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from models.detector import create_model

# Create a model with predefined configuration
model = create_model(model_config='base', num_classes=80)

# Forward pass
import torch
x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
output = model(x)
```

### Using Custom Configuration

```python
from models.detector import ObjectDetectionModel
from utils.config import Config

# Load custom configuration
config = Config('configs/my_custom_config.yaml')
model = ObjectDetectionModel(config_path='configs/my_custom_config.yaml')
```

### Available Model Configurations

- **tiny**: Minimal parameters for fast training/inference
- **small**: Balanced parameters for good performance
- **base**: Standard parameters for production use
- **large**: Maximum parameters for best accuracy

## Configuration

Model parameters are managed through YAML configuration files in the `configs/` directory. Each configuration includes:

- **enhancement**: ViT and ResNeXt parameters
- **depth**: Swin Transformer parameters
- **evolution**: Encoder-decoder parameters
- **detection**: Head and output parameters
- **training**: Training hyperparameters

### Example Configuration

```yaml
model:
  name: "base"
  input_size: [224, 224]
  num_classes: 80

enhancement:
  embed_dim: 64
  num_heads: 8
  num_blocks: 3
  window_size: 7
  cardinality: 32
  resnext_type: "C"

depth:
  in_channels: 64
  embed_dim: 96
  out_channels: 128
  depth: 3
  num_heads: 8

# ... more configuration sections
```

## Model Components

### Enhancement Module
- Combines Swin Transformer attention with ResNeXt grouped convolutions
- Supports both ResNeXt Form B (split-transform-merge) and Form C (grouped convolution)
- Includes residual connections for stable training

### Depth Module
- Sequential Swin Transformer blocks with alternating window attention
- Supports both sequence and image format inputs
- Includes projection layers for better feature transformation

### Feature Evolution Subnetwork
- **Transformation Module**: 3-layer encoder with stride-2 convolutions
- **Recovery Module**: 3-layer decoder with transpose convolutions
- Supports separate encoding and decoding for feature extraction

### Detection Head
- Multi-scale detection with separate heads for different feature levels
- Outputs class predictions, bounding box regressions, and objectness scores

## Training

The model supports training with custom datasets. Configuration files include training parameters such as:

- Batch size
- Learning rate
- Number of epochs
- Weight decay
- Momentum

## Testing

Run the test script to verify model functionality:

```bash
python models/detector.py
```

This will test all model configurations and output parameter counts and tensor shapes.

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.6.0
- PyYAML >= 6.0
- numpy >= 1.21.0
- OpenCV >= 4.5.0

## Features

- ✅ Modular architecture with separate components
- ✅ YAML-based configuration management
- ✅ Multiple model sizes (tiny, small, base, large)
- ✅ Low-light image enhancement
- ✅ Multi-scale feature processing
- ✅ Comprehensive documentation
- ✅ Easy to extend and customize

## Future Development

- [ ] Add training and evaluation scripts
- [ ] Implement data augmentation
- [ ] Add model checkpointing
- [ ] Support for different backbone architectures
- [ ] Integration with popular datasets (COCO, VOC)
- [ ] Model quantization for deployment
- [ ] ONNX export support

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
# diffnet
