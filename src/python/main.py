"""
CrowdFace Main Module

This module provides the entry point for the CrowdFace system,
integrating SAM2, RVM, and BAGEL for neural-adaptive crowd segmentation
with contextual pixel-space advertisement integration.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.crowdface_pipeline import CrowdFacePipeline
from python.bagel_integration import setup_bagel_integration
from python.utils import load_video, create_sample_ad, save_video, display_comparison

def load_sam_model(model_path=None):
    """
    Load the SAM2 model.
    
    Args:
        model_path: Path to the model weights (optional)
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import SamModel, SamProcessor
        
        # Use default model ID if path not provided
        model_id = model_path or "facebook/sam2"
        logger.info(f"Loading SAM2 model from {model_id}")
        
        # Try to get token from environment
        token = os.environ.get('HUGGINGFACE_TOKEN')
        
        # Load processor and model
        sam_processor = SamProcessor.from_pretrained(model_id, token=token)
        sam_model = SamModel.from_pretrained(model_id, token=token)
        
        # Move model to appropriate device
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam_model = sam_model.to(device)
        
        logger.info("SAM2 model loaded successfully")
        return sam_model, sam_processor
        
    except Exception as e:
        logger.error(f"Error loading SAM2 model: {e}")
        logger.warning("Will use a placeholder for demonstration purposes")
        return None, None

def load_rvm_model(model_path=None):
    """
    Load the RVM model.
    
    Args:
        model_path: Path to the model weights (optional)
        
    Returns:
        RVM model
    """
    try:
        # Try to import RVM
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'RobustVideoMatting'))
        from model import MattingNetwork
        import torch
        
        # Use default path if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'rvm_mobilenetv3.pth')
        
        logger.info(f"Loading RVM model from {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"RVM model file not found: {model_path}")
            return None
        
        # Load RVM model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rvm_model = MattingNetwork('mobilenetv3').eval().to(device)
        
        # Load weights
        rvm_model.load_state_dict(torch.load(model_path, map_location=device))
        
        logger.info("RVM model loaded successfully")
        return rvm_model
        
    except Exception as e:
        logger.error(f"Error loading RVM model: {e}")
        logger.warning("Will use a placeholder for demonstration purposes")
        return None

def main():
    """
    Main entry point for the CrowdFace system.
    """
    parser = argparse.ArgumentParser(description='CrowdFace: Neural-Adaptive Crowd Segmentation with Ad Integration')
    parser.add_argument('--input', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--ad', type=str, help='Advertisement image path')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum number of frames to process')
    parser.add_argument('--scale', type=float, default=0.3, help='Scale factor for the ad (0.0-1.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-bagel', action='store_true', help='Disable BAGEL integration')
    parser.add_argument('--bagel-path', type=str, help='Path to BAGEL repository')
    parser.add_argument('--bagel-model', type=str, help='Path to BAGEL model')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load video
    if args.input:
        video_path = args.input
    else:
        # Use sample video if no input provided
        video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sample_video.mp4')
        if not os.path.exists(video_path):
            logger.error(f"No input video provided and sample video not found at {video_path}")
            return 1
    
    frames = load_video(video_path, max_frames=args.max_frames)
    if not frames:
        logger.error("Failed to load video frames")
        return 1
    
    # Load or create ad image
    ad_image = None
    if args.ad:
        try:
            ad_image = cv2.imread(args.ad, cv2.IMREAD_UNCHANGED)
            if ad_image is None:
                logger.error(f"Failed to load ad image from {args.ad}")
                ad_image = None
        except Exception as e:
            logger.error(f"Error loading ad image: {e}")
            ad_image = None
    
    # Load models
    sam_model, sam_processor = load_sam_model()
    rvm_model = load_rvm_model()
    
    # Set up BAGEL integration
    bagel_integration = None
    if not args.no_bagel:
        bagel_integration = setup_bagel_integration(
            bagel_path=args.bagel_path,
            use_gpu=torch.cuda.is_available(),
            model_path=args.bagel_model
        )
    
    # Initialize pipeline
    pipeline = CrowdFacePipeline(
        sam_model=sam_model,
        sam_processor=sam_processor,
        rvm_model=rvm_model,
        bagel_integration=bagel_integration
    )
    
    # Process video
    output_path = args.output
    processed_frames = pipeline.process_video(
        frames,
        ad_image,
        output_path=output_path
    )
    
    if not processed_frames:
        logger.error("Failed to process video")
        return 1
    
    logger.info(f"Successfully processed {len(processed_frames)} frames")
    logger.info(f"Output saved to {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
