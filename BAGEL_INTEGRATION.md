# BAGEL Integration in CrowdFace

This document describes the integration of ByteDance's BAGEL (ByteDance Ad Generation and Embedding Library) into the CrowdFace system.

## Overview

BAGEL is an advanced AI system developed by ByteDance that provides intelligent ad placement capabilities. In CrowdFace, BAGEL is used to analyze video frames and determine optimal locations for ad placement based on scene understanding and content analysis.

## Integration Architecture

The integration follows these key principles:

1. **Loose Coupling**: CrowdFace can function without BAGEL, falling back to basic placement algorithms
2. **Seamless Enhancement**: When BAGEL is available, it enhances ad placement with advanced features
3. **Consistent API**: The same API is used regardless of whether BAGEL is available

## Setup Instructions

### 1. Clone the BAGEL Repository

```bash
git clone https://github.com/ByteDance-Seed/Bagel.git
```

The repository should be cloned into the root directory of the CrowdFace project.

### 2. Install BAGEL Dependencies

```bash
cd Bagel
pip install -r requirements.txt
```

### 3. Set Environment Variables

For Hugging Face model access:

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Usage

The BAGEL integration is handled through the `BAGELWrapper` class in `src/python/bagel_loader.py`. This wrapper provides:

1. **Model Loading**: Handles loading the BAGEL models with appropriate error handling
2. **Frame Analysis**: Processes video frames to determine optimal ad placement
3. **Fallback Mechanisms**: Provides basic functionality when BAGEL is unavailable

## Key Features

When integrated with BAGEL, CrowdFace gains these additional capabilities:

### Scene Understanding

BAGEL analyzes the video content to understand:
- Scene type (indoor/outdoor, crowd density, etc.)
- Visual context and mood
- Audience demographics

### Intelligent Ad Placement

Based on scene analysis, BAGEL determines:
- Optimal ad placement locations
- Appropriate ad sizes and styles
- Contextual relevance scoring

### Ad Effectiveness Prediction

BAGEL can predict:
- Viewer attention patterns
- Ad visibility metrics
- Potential engagement levels

## Implementation Details

The integration is implemented in three main components:

1. **BAGELWrapper** (`src/python/bagel_loader.py`): Handles loading and initializing BAGEL
2. **CrowdFacePipeline** (`src/python/crowdface_pipeline.py`): Uses BAGEL for ad placement
3. **Main Module** (`src/python/main.py`): Orchestrates the integration

## Fallback Mechanism

When BAGEL is unavailable, CrowdFace falls back to a basic placement algorithm that:
1. Identifies people in the frame using segmentation masks
2. Places ads in empty spaces, typically to the right of detected people
3. Ensures ads don't overlap with important content

## References

- [BAGEL GitHub Repository](https://github.com/ByteDance-Seed/Bagel)
- [ByteDance Research](https://bytedance.com/en/research)
