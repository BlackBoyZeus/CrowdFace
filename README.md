# CrowdFace

Neural-Adaptive Crowd Segmentation with Contextual Pixel-Space Advertisement Integration

## Overview

CrowdFace is an advanced computer vision system that combines state-of-the-art segmentation models with contextual advertisement placement. The system is designed to identify and analyze crowd scenes, segment individuals, and intelligently place advertisements in appropriate spaces within the video frame.

## Features

- **Advanced Crowd Segmentation**: Uses SAM2 (Segment Anything Model 2) for precise crowd detection and segmentation
- **Robust Video Matting**: Implements RVM (Robust Video Matting) for high-quality alpha matte generation
- **Contextual Ad Placement**: BAGEL (ByteDance Ad Generation and Embedding Library) for intelligent ad placement
- **Multi-platform Support**: Works on various operating systems with GPU acceleration
- **AWS Integration**: Seamless deployment and scaling on AWS infrastructure

## Demo Notebook

Try our interactive demo notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlackBoyZeus/CrowdFace/blob/main/CrowdFace_Demo.ipynb)

The notebook demonstrates the complete pipeline including:
- Loading SAM2, RVM, and BAGEL models
- Processing uploaded videos
- Intelligent ad placement with scene understanding
- Contextual ad optimization

## Architecture

The system consists of three main components:

1. **Segmentation Module**: Identifies and segments people in crowd scenes using SAM2
2. **Matting Module**: Creates precise alpha mattes for seamless integration using RVM
3. **Injection Module**: Places advertisements contextually within the scene using BAGEL

### BAGEL Integration

The BAGEL (ByteDance Ad Generation and Embedding Library) integration provides:
- Scene understanding and context analysis
- Intelligent ad placement based on scene content
- Ad content optimization to match scene mood and context

For more details, see [BAGEL_INTEGRATION.md](BAGEL_INTEGRATION.md).

## Requirements

### Rust Dependencies
- Rust 1.70+
- OpenCV 4.x
- PyTorch (libtorch)

### Python Dependencies
- Python 3.10+
- PyTorch
- OpenCV
- Transformers
- Diffusers
- Accelerate
- Hugging Face Hub

### System Requirements
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 100GB+ disk space

## Installation

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/BlackBoyZeus/CrowdFace.git
cd CrowdFace

# Install Rust dependencies
cd src/rust
cargo build

# Install Python dependencies
cd ../python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run the server
cargo run --bin crowdface-server -- --config config.json

# Process a video file
python src/python/process_video.py --input video.mp4 --output output.mp4 --ad ad_image.png
```

### Using the Python API

```python
from src.python.bagel_loader import load_bagel_model
from src.python.crowdface_pipeline import CrowdFacePipeline

# Load models
bagel_model, bagel_inferencer = load_bagel_model("models/BAGEL-7B-MoT")

# Initialize pipeline
pipeline = CrowdFacePipeline(
    sam_model=sam_model,
    sam_processor=sam_processor,
    rvm_model=rvm_model,
    bagel_inferencer=bagel_inferencer
)

# Process video
pipeline.process_video(
    video_path="input.mp4",
    ad_image="ad.png",
    output_path="output.mp4"
)
```

## License

Commercial use license required. Contact for licensing terms.

## Contributors

- Development Team at CrowdFace Technologies
