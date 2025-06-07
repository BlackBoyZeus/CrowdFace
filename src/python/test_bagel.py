#!/usr/bin/env python3
"""
Simple test script for the BAGEL API
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Mock the imports from bagel_api to avoid loading the actual models
class BagelConfig:
    """Configuration for BAGEL model."""
    def __init__(self, model_path="mock/model", device="cpu", context_awareness=0.7, 
                 cfg_scale=7.0, seed=None, num_inference_steps=30, 
                 image_guidance_scale=1.5, use_fp16=True, cache_dir=None):
        self.model_path = model_path
        self.device = device
        self.context_awareness = context_awareness
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir
    
    def __str__(self):
        return f"BagelConfig(model_path={self.model_path}, device={self.device}, use_fp16={self.use_fp16})"


class AdPlacementOptions:
    """Options for ad placement."""
    def __init__(self, opacity=0.9, blend_mode="normal", size_ratio=0.2, 
                 position_constraints=None, avoid_faces=True, avoid_text=True, 
                 temporal_consistency=0.8):
        self.opacity = opacity
        self.blend_mode = blend_mode
        self.size_ratio = size_ratio
        self.position_constraints = position_constraints
        self.avoid_faces = avoid_faces
        self.avoid_text = avoid_text
        self.temporal_consistency = temporal_consistency
    
    def __str__(self):
        return f"AdPlacementOptions(opacity={self.opacity}, blend_mode={self.blend_mode}, size_ratio={self.size_ratio})"


class AdContent:
    """Ad content for injection."""
    def __init__(self, image, target_width=None, target_height=None, mask=None, metadata=None):
        self.image = image
        self.target_width = target_width
        self.target_height = target_height
        self.mask = mask
        self.metadata = metadata or {}


class PlacementResult:
    """Result of ad placement."""
    def __init__(self, position, size, rotation, opacity, context_score, composited_frame):
        self.position = position
        self.size = size
        self.rotation = rotation
        self.opacity = opacity
        self.context_score = context_score
        self.composited_frame = composited_frame


class MockBagelModel:
    """Mock implementation of BagelModel for testing"""
    def __init__(self, config):
        self.config = config
        print(f"Initialized mock BAGEL model with config: {config}")
    
    def place_ad(self, frame, ad_content, options=None):
        options = options or AdPlacementOptions()
        print(f"Mock placing ad with options: {options}")
        
        # Create a simple result by overlaying the ad on the frame
        result_frame = frame.copy()
        
        # Get ad as numpy array
        if isinstance(ad_content.image, str):
            ad_img = cv2.imread(ad_content.image, cv2.IMREAD_UNCHANGED)
        elif isinstance(ad_content.image, np.ndarray):
            ad_img = ad_content.image
        else:
            ad_img = np.array(ad_content.image)
        
        # Simple placement in the center
        h, w = ad_img.shape[:2]
        y = (result_frame.shape[0] - h) // 2
        x = (result_frame.shape[1] - w) // 2
        
        # Simple alpha blending if we have alpha channel
        if ad_img.shape[2] == 4:
            alpha = ad_img[:, :, 3] / 255.0 * options.opacity
            for c in range(3):
                result_frame[y:y+h, x:x+w, c] = (
                    result_frame[y:y+h, x:x+w, c] * (1 - alpha) + 
                    ad_img[:, :, c] * alpha
                )
        else:
            # Just copy the RGB channels
            result_frame[y:y+h, x:x+w] = ad_img[:, :, :3]
        
        return PlacementResult(
            position=(x, y),
            size=(w, h),
            rotation=0.0,
            opacity=options.opacity,
            context_score=0.85,
            composited_frame=result_frame
        )


def main():
    print("Testing BAGEL API with a sample image")
    
    # Use a sample image from skimage
    try:
        from skimage import data
        sample_image = data.astronaut()
        print(f"Loaded sample image with shape: {sample_image.shape}")
    except ImportError:
        print("Could not import skimage. Using a simple test image instead.")
        # Create a simple test image
        sample_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add some color gradient
        for i in range(512):
            for j in range(512):
                sample_image[i, j, 0] = i // 2  # Red
                sample_image[i, j, 1] = j // 2  # Green
                sample_image[i, j, 2] = 128     # Blue
        print(f"Created test image with shape: {sample_image.shape}")
    
    # Create a simple ad image
    ad_image = np.zeros((100, 200, 4), dtype=np.uint8)
    ad_image[:, :, 0] = 255  # Red
    ad_image[:, :, 3] = 200  # Alpha
    
    # Save the test images
    cv2.imwrite("test_frame.jpg", cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_ad.png", ad_image)
    
    print("Test images saved as test_frame.jpg and test_ad.png")
    
    # Initialize the model with CPU configuration
    print("Initializing mock BAGEL model")
    
    config = BagelConfig(
        model_path="mock/bagel-model",
        device="cpu",
        use_fp16=False
    )
    
    model = MockBagelModel(config)
    
    # Load the test images
    frame = cv2.imread("test_frame.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    ad_content = AdContent(image="test_ad.png")
    
    options = AdPlacementOptions(
        opacity=0.9,
        blend_mode="normal",
        size_ratio=0.2
    )
    
    # Place ad
    print("Placing ad in frame...")
    result = model.place_ad(frame, ad_content, options)
    
    # Save result
    print(f"Ad placed at position {result.position} with context score {result.context_score}")
    cv2.imwrite("result.jpg", cv2.cvtColor(result.composited_frame, cv2.COLOR_RGB2BGR))
    
    print("Result saved as result.jpg")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
