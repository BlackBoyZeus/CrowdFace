"""
CrowdFace Pipeline Implementation

This module provides the core functionality for the CrowdFace system,
which combines SAM2 (Segment Anything Model 2), RVM (Robust Video Matting),
and BAGEL (ByteDance Ad Generation and Embedding Library) for neural-adaptive
crowd segmentation with contextual pixel-space advertisement integration.
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrowdFacePipeline:
    """
    Main pipeline for CrowdFace system that handles segmentation, matting,
    and ad placement in videos.
    """
    
    def __init__(self, sam_model=None, sam_processor=None, rvm_model=None, bagel_integration=None):
        """
        Initialize the CrowdFace pipeline with optional models.
        
        Args:
            sam_model: SAM2 model for segmentation
            sam_processor: SAM2 processor for input preparation
            rvm_model: RVM model for video matting
            bagel_integration: BAGEL integration for ad placement optimization
        """
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        self.rvm_model = rvm_model
        self.bagel_integration = bagel_integration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state variables for video processing
        self.prev_frame = None
        self.prev_fgr = None
        self.prev_pha = None
        self.prev_state = None
        
        logger.info(f"CrowdFace pipeline initialized with device: {self.device}")
        logger.info(f"SAM2 model: {'Loaded' if sam_model else 'Not loaded'}")
        logger.info(f"RVM model: {'Loaded' if rvm_model else 'Not loaded'}")
        logger.info(f"BAGEL integration: {'Available' if bagel_integration else 'Not available'}")
    
    def segment_people(self, frame):
        """
        Segment people in the frame using SAM2 or fallback to a placeholder.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            Binary mask of segmented people (numpy array)
        """
        if self.sam_model is None or self.sam_processor is None:
            # Create a simple placeholder mask for demonstration
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            # Add a simple ellipse as a "person"
            cv2.ellipse(mask, 
                       (frame.shape[1]//2, frame.shape[0]//2),
                       (frame.shape[1]//4, frame.shape[0]//2),
                       0, 0, 360, 255, -1)
            return mask
            
        # Convert frame to RGB if it's in BGR format
        if isinstance(frame, np.ndarray) and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        # Process the image with SAM
        inputs = self.sam_processor(rgb_frame, return_tensors="pt").to(self.device)
        
        # Generate automatic mask predictions
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
            
        # Get the predicted masks
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Take the largest mask as a person (simplified approach)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        if len(masks) > 0 and len(masks[0]) > 0:
            largest_mask = None
            largest_area = 0
            
            for mask in masks[0]:
                mask_np = mask.numpy()
                area = np.sum(mask_np)
                if area > largest_area:
                    largest_area = area
                    largest_mask = mask_np
                    
            if largest_mask is not None:
                combined_mask = largest_mask.astype(np.uint8) * 255
        
        return combined_mask
    
    def generate_matte(self, frame):
        """
        Generate alpha matte using RVM or fallback to segmentation.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            Alpha matte (numpy array)
        """
        if self.rvm_model is None:
            # Fallback to simple segmentation
            return self.segment_people(frame)
            
        try:
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Initialize previous frame and state if not provided
            if self.prev_frame is None:
                self.prev_frame = torch.zeros_like(frame_tensor)
            if self.prev_fgr is None:
                self.prev_fgr = torch.zeros_like(frame_tensor)
            if self.prev_pha is None:
                self.prev_pha = torch.zeros((1, 1, frame.shape[0], frame.shape[1]), device=self.device)
                
            # Generate matte
            with torch.no_grad():
                fgr, pha, state = self.rvm_model(frame_tensor, self.prev_frame, self.prev_fgr, self.prev_pha, self.prev_state)
                
            # Update state for next frame
            self.prev_frame = frame_tensor
            self.prev_fgr = fgr
            self.prev_pha = pha
            self.prev_state = state
                
            # Convert alpha matte to numpy array
            alpha_matte = pha[0, 0].cpu().numpy() * 255
            alpha_matte = alpha_matte.astype(np.uint8)
            
            return alpha_matte
            
        except Exception as e:
            logger.error(f"Error in RVM matting: {e}")
            # Fallback to segmentation mask
            return self.segment_people(frame)
    
    def find_ad_placement(self, frame, mask):
        """
        Find suitable locations for ad placement based on segmentation.
        
        Args:
            frame: Input video frame (numpy array)
            mask: Segmentation mask (numpy array)
            
        Returns:
            (x, y) coordinates for ad placement
        """
        # Use BAGEL if available for optimal placement
        if self.bagel_integration is not None:
            try:
                # Get BAGEL placement recommendations
                placement_info = self.bagel_integration.find_optimal_placement(frame, mask)
                
                # Extract optimal placement
                if 'optimal_placement' in placement_info:
                    logger.info(f"Using BAGEL placement: {placement_info['optimal_placement']}")
                    return placement_info['optimal_placement']
            except Exception as e:
                logger.error(f"Error in BAGEL ad placement: {e}")
                # Fall back to basic placement
        
        # Basic placement logic
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Default to center-right if no contours found
            return (frame.shape[1] * 3 // 4, frame.shape[0] // 2)
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Default placement to the right of the person
        ad_x = min(x + w + 20, frame.shape[1] - 100)
        ad_y = y
        
        return (ad_x, ad_y)
    
    def place_ad(self, frame, ad_image, position, scale=0.3):
        """
        Place the ad in the frame at the specified position with alpha blending.
        
        Args:
            frame: Input video frame (numpy array)
            ad_image: Advertisement image with alpha channel (numpy array or PIL Image)
            position: (x, y) coordinates for placement
            scale: Scale factor for the ad image (0.0-1.0)
            
        Returns:
            Frame with ad placed (numpy array)
        """
        # Convert ad_image to numpy array if it's a PIL Image
        if isinstance(ad_image, Image.Image):
            ad_image = np.array(ad_image)
            # Convert RGB to BGR if needed
            if ad_image.shape[2] == 3:
                ad_image = cv2.cvtColor(ad_image, cv2.COLOR_RGB2BGR)
        
        # Resize ad image
        ad_height = int(frame.shape[0] * scale)
        ad_width = int(ad_image.shape[1] * (ad_height / ad_image.shape[0]))
        ad_resized = cv2.resize(ad_image, (ad_width, ad_height))
        
        # Extract position
        x, y = position
        
        # Ensure the ad fits within the frame
        if x + ad_width > frame.shape[1]:
            x = frame.shape[1] - ad_width
        if y + ad_height > frame.shape[0]:
            y = frame.shape[0] - ad_height
            
        # Create a copy of the frame
        result = frame.copy()
        
        # Check if ad has an alpha channel
        if ad_resized.shape[2] == 4:
            # Extract alpha channel
            alpha = ad_resized[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            # Extract RGB channels
            rgb = ad_resized[:, :, :3]
            
            # Get the region of interest in the frame
            roi = result[y:y+ad_height, x:x+ad_width]
            
            # Blend the ad with the frame using alpha
            blended = (1.0 - alpha) * roi + alpha * rgb
            
            # Place the blended image back into the frame
            result[y:y+ad_height, x:x+ad_width] = blended
        else:
            # Simple overlay without alpha blending
            result[y:y+ad_height, x:x+ad_width] = ad_resized
            
        return result
    
    def process_video(self, frames, ad_image=None, output_path=None, display_results=True):
        """
        Process video frames with ad placement.
        
        Args:
            frames: List of video frames (numpy arrays)
            ad_image: Advertisement image with alpha channel (numpy array or PIL Image)
            output_path: Path to save the output video (optional)
            display_results: Whether to display comparison results (boolean)
            
        Returns:
            List of processed frames (numpy arrays)
        """
        # Process video frames with ad placement
        results = []
        
        # Check if frames list is empty
        if not frames:
            logger.error("No frames to process")
            return results
        
        # Generate contextual ad if not provided and BAGEL is available
        if ad_image is None and self.bagel_integration is not None:
            try:
                logger.info("Generating contextual ad using BAGEL...")
                ad_image = self.bagel_integration.generate_contextual_ad(frames[0])
                logger.info("Ad generated successfully")
            except Exception as e:
                logger.error(f"Error generating ad: {e}")
                # Create a default ad
                ad_image = self._create_default_ad()
        elif ad_image is None:
            # Create a default ad
            ad_image = self._create_default_ad()
        
        # Reset state variables
        self.prev_frame = None
        self.prev_fgr = None
        self.prev_pha = None
        self.prev_state = None
        
        logger.info(f"Processing {len(frames)} frames")
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Every 10 frames, re-detect people and ad placement
            if i % 10 == 0:
                mask = self.generate_matte(frame)
                ad_position = self.find_ad_placement(frame, mask)
                logger.debug(f"Frame {i}: Ad position = {ad_position}")
            
            # Place the ad
            result_frame = self.place_ad(frame, ad_image, ad_position)
            results.append(result_frame)
            
        # Save video if output path is provided
        if output_path and results:
            height, width = results[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
            
            for frame in results:
                out.write(frame)
                
            out.release()
            logger.info(f"Video saved to {output_path}")
            
        return results
        
    def _create_default_ad(self):
        """
        Create a default advertisement image.
        
        Returns:
            Ad image with alpha channel
        """
        # Create a sample advertisement image with transparency
        ad_width, ad_height = 500, 300
        ad_img = np.zeros((ad_height, ad_width, 4), dtype=np.uint8)
        
        # Create a semi-transparent rectangle
        cv2.rectangle(ad_img, (25, 25), (ad_width-25, ad_height-25), (0, 120, 255, 180), -1)
        cv2.rectangle(ad_img, (25, 25), (ad_width-25, ad_height-25), (0, 0, 0, 255), 3)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ad_img, "CROWDFACE", (50, 100), font, 2, (255, 255, 255, 255), 5)
        cv2.putText(ad_img, "DEMO AD", (120, 200), font, 1.5, (255, 255, 255, 255), 3)
        
        return ad_img
