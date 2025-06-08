"""
CrowdFacePipeline - Main pipeline for CrowdFace system
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any

class CrowdFacePipeline:
    def __init__(self, sam_model, sam_processor, rvm_model, bagel_inferencer=None):
        """
        Initialize the CrowdFace pipeline
        
        Args:
            sam_model: SAM2 model for segmentation
            sam_processor: SAM2 processor
            rvm_model: RVM model for matting
            bagel_inferencer: BAGEL model inferencer (optional)
        """
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        self.rvm_model = rvm_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Import BAGEL components only if inferencer is provided
        if bagel_inferencer is not None:
            from bagel import BAGELSceneUnderstanding, BAGELAdPlacement, BAGELAdOptimization
            self.scene_analyzer = BAGELSceneUnderstanding(bagel_inferencer)
            self.ad_placer = BAGELAdPlacement(bagel_inferencer)
            self.ad_optimizer = BAGELAdOptimization(bagel_inferencer)
        else:
            self.scene_analyzer = None
            self.ad_placer = None
            self.ad_optimizer = None
        
    def segment_people(self, frame):
        """
        Segment people in the frame using SAM2
        
        Args:
            frame: The video frame
            
        Returns:
            Segmentation mask
        """
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
        
        # Filter masks to only include people (this is a simplified approach)
        # In a real implementation, you would use a classifier to identify people
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Take the largest mask as a person (simplified approach)
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
        
    def generate_matte(self, frame, prev_frame=None, prev_fgr=None, prev_pha=None, prev_state=None):
        """
        Generate alpha matte using RVM
        
        Args:
            frame: The video frame
            prev_frame: Previous frame tensor
            prev_fgr: Previous foreground tensor
            prev_pha: Previous alpha tensor
            prev_state: Previous RVM state
            
        Returns:
            Tuple of (alpha_matte, foreground, alpha, state)
        """
        if self.rvm_model is None:
            # Fallback to simple mask if RVM is not available
            return self.segment_people(frame)
            
        try:
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Initialize previous frame and state if not provided
            if prev_frame is None:
                prev_frame = torch.zeros_like(frame_tensor)
            if prev_fgr is None:
                prev_fgr = torch.zeros_like(frame_tensor)
            if prev_pha is None:
                prev_pha = torch.zeros((1, 1, frame.shape[0], frame.shape[1]), device=self.device)
            if prev_state is None:
                prev_state = None
                
            # Generate matte
            with torch.no_grad():
                fgr, pha, state = self.rvm_model(frame_tensor, prev_frame, prev_fgr, prev_pha, prev_state)
                
            # Convert alpha matte to numpy array
            alpha_matte = pha[0, 0].cpu().numpy() * 255
            alpha_matte = alpha_matte.astype(np.uint8)
            
            return alpha_matte, fgr, pha, state
            
        except Exception as e:
            print(f"Error in RVM matting: {e}")
            # Fallback to segmentation mask
            return self.segment_people(frame), None, None, None
        
    def find_ad_placement(self, frame, mask, scene_info=None):
        """
        Find suitable locations for ad placement
        
        Args:
            frame: The video frame
            mask: Segmentation mask
            scene_info: Scene analysis information
            
        Returns:
            Tuple of (x, y) coordinates for ad placement
        """
        # Use BAGEL-enhanced ad placement if available
        if self.ad_placer is not None and scene_info is not None:
            return self.ad_placer.find_optimal_placement(frame, mask, scene_info)
        
        # Fall back to original method if BAGEL is not available
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Default placement to the right of the person
        ad_x = x + w + 20
        ad_y = y
        
        return (ad_x, ad_y)
        
    def optimize_ad(self, ad_image, scene_info=None):
        """
        Optimize ad content based on scene context
        
        Args:
            ad_image: The advertisement image
            scene_info: Scene analysis information
            
        Returns:
            Optimized advertisement image
        """
        if self.ad_optimizer is not None and scene_info is not None:
            return self.ad_optimizer.optimize_ad_for_scene(ad_image, scene_info)
        return ad_image
        
    def place_ad(self, frame, ad_image, position, scale=0.5, scene_info=None):
        """
        Place the ad in the frame at the specified position
        
        Args:
            frame: The video frame
            ad_image: The advertisement image
            position: (x, y) coordinates for placement
            scale: Size scale factor for the ad
            scene_info: Scene analysis information
            
        Returns:
            Frame with ad placed
        """
        # Optimize ad content if scene info is available
        if scene_info is not None and self.ad_optimizer is not None:
            ad_image = self.optimize_ad(ad_image, scene_info)
            
        # Determine optimal scale based on scene context
        if scene_info is not None and "description" in scene_info:
            if "crowded" in scene_info["description"].lower():
                scale = 0.3  # Smaller ads in crowded scenes
            elif "spacious" in scene_info["description"].lower():
                scale = 0.7  # Larger ads in spacious scenes
        
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
        
    def process_video(self, video_path, ad_image, output_path, max_frames=None):
        """
        Process the entire video with BAGEL-enhanced intelligence
        
        Args:
            video_path: Path to input video
            ad_image: Advertisement image or path
            output_path: Path for output video
            max_frames: Maximum number of frames to process
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
            
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load ad image
        if isinstance(ad_image, str):
            ad_img = cv2.imread(ad_image, cv2.IMREAD_UNCHANGED)
        else:
            ad_img = ad_image
            
        # Process each frame
        frame_idx = 0
        ad_position = None
        scene_info = None
        prev_fgr, prev_pha, prev_state = None, None, None
        
        for _ in tqdm(range(frame_count), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Every 30 frames, analyze scene and re-detect people and ad placement
            if frame_idx % 30 == 0:
                # Analyze scene with BAGEL if available
                if self.scene_analyzer is not None:
                    scene_info = self.scene_analyzer.analyze_frame(frame, frame_idx)
                    print(f"Scene context: {scene_info['context']}")
                    print(f"Recommended ad types: {scene_info['suitable_ad_types']}")
                
                # Generate matte
                if self.rvm_model is not None:
                    mask, prev_fgr, prev_pha, prev_state = self.generate_matte(frame)
                else:
                    mask = self.segment_people(frame)
                
                # Find ad placement using scene information
                ad_position = self.find_ad_placement(frame, mask, scene_info)
            
            # For frames between key frames, update the matte with RVM
            elif self.rvm_model is not None and prev_pha is not None:
                prev_frame = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                prev_frame = prev_frame.to(self.device)
                mask, prev_fgr, prev_pha, prev_state = self.generate_matte(
                    frame, prev_frame, prev_fgr, prev_pha, prev_state
                )
            
            # If we found a position for the ad, place it
            if ad_position is not None:
                frame = self.place_ad(frame, ad_img, ad_position, scene_info=scene_info)
            
            # Write the frame to the output video
            out.write(frame)
            frame_idx += 1
            
        # Release resources
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")
