# BAGEL Integration for CrowdFace

This document describes the integration of BAGEL (ByteDance Ad Generation and Embedding Library) into the CrowdFace system.

## Overview

BAGEL enhances CrowdFace with intelligent scene understanding and contextual ad placement capabilities. The integration consists of three main components:

1. **Scene Understanding**: Analyzes video frames to extract context, mood, and objects
2. **Ad Placement**: Determines optimal ad placement based on scene context
3. **Ad Optimization**: Modifies ad content to better match the scene context

## Implementation

The BAGEL integration is implemented in the following files:

- `src/python/bagel/scene_understanding.py`: Scene analysis using BAGEL model
- `src/python/bagel/ad_placement.py`: Intelligent ad placement
- `src/python/bagel/ad_optimization.py`: Context-aware ad content optimization
- `src/python/bagel_loader.py`: Utility for loading BAGEL model
- `src/python/crowdface_pipeline.py`: Main pipeline integrating all components

## Usage

```python
# Import required modules
from src.python.bagel_loader import load_bagel_model
from src.python.crowdface_pipeline import CrowdFacePipeline

# Load BAGEL model
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

## Key Features

### Scene Understanding

The `BAGELSceneUnderstanding` class analyzes video frames to extract:

- **Context**: Outdoor/indoor, crowded/spacious, urban/rural, etc.
- **Objects**: People, cars, buildings, etc.
- **Mood**: Happy, serious, relaxed, energetic, tense
- **Ad Type Recommendations**: Based on scene context

### Ad Placement

The `BAGELAdPlacement` class determines optimal ad placement:

- Analyzes scene context to decide where ads should appear
- Places ads in appropriate locations (right, left, top, bottom)
- Avoids placing ads over important scene elements

### Ad Optimization

The `BAGELAdOptimization` class modifies ad content:

- Adjusts ad style to match scene mood
- Optimizes ad content based on scene context
- Caches optimized ads for efficiency

## Notebook Demo

A Jupyter notebook demonstrating the BAGEL integration is available at:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlackBoyZeus/CrowdFace/blob/main/CrowdFace_Demo.ipynb)

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Transformers
- Accelerate
- Hugging Face Hub
- BAGEL model weights (available from ByteDance)
