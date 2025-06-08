"""
CrowdFace Python API
Neural-Adaptive Crowd Segmentation with Contextual Pixel-Space Advertisement Integration
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union


class CrowdFacePipeline:
    """Main pipeline for CrowdFace video processing with ad integration"""
    
    def __init__(
        self,
        sam_model,
        sam_processor,
        rvm_model,
        bagel_inferencer=None
    ):
        """
        Initialize the CrowdFace pipeline
        
        Args:
            sam_model: SAM2 model for segmentation
            sam_processor: SAM2 processor for image preprocessing
            rvm_model: RVM model for video matting
            bagel_inferencer: BAGEL model for ad placement (optional)
        """
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        self.rvm_model = rvm_model
        self.bagel_inferencer = bagel_inferencer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state variables for video processing
        self.prev_frame = None
        self.prev_fgr = None
        self.prev_pha = None
        self.prev_state = None
    
    def segment_people(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment people in the frame using SAM2
        
        Args:
            frame: Input video frame
            
        Returns:
            Binary mask of people in the frame
        """
        if self.sam_model is None or self.sam_processor is None:
            # Create a simple placeholder mask for demonstration
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            # Add a simple ellipse as a "person"
            cv2.ellipse(
                mask, 
                (frame.shape[1]//2, frame.shape[0]//2),
                (frame.shape[1]//4, frame.shape[0]//2),
                0, 0, 360, 255, -1
            )
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
    
    def generate_matte(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate alpha matte using RVM
        
        Args:
            frame: Input video frame
            
        Returns:
            Alpha matte for the frame
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
                fgr, pha, state = self.rvm_model(
                    frame_tensor, 
                    self.prev_frame, 
                    self.prev_fgr, 
                    self.prev_pha, 
                    self.prev_state
                )
                
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
            print(f"Error in RVM matting: {e}")
            # Fallback to segmentation mask
            return self.segment_people(frame)
    
    def find_ad_placement(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
        """
        Find suitable locations for ad placement
        
        Args:
            frame: Input video frame
            mask: Binary mask of people in the frame
            
        Returns:
            (x, y) coordinates for ad placement
        """
        # Use BAGEL if available
        if self.bagel_inferencer is not None:
            try:
                # This would be the actual BAGEL implementation
                # For now, we'll use a placeholder
                scene_analysis = self.bagel_inferencer.analyze_scene(frame)
                return scene_analysis.get_optimal_placement()
            except Exception as e:
                print(f"Error in BAGEL ad placement: {e}")
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
    
    def place_ad(
        self, 
        frame: np.ndarray, 
        ad_image: Union[np.ndarray, Image.Image], 
        position: Tuple[int, int], 
        scale: float = 0.3
    ) -> np.ndarray:
        """
        Place the ad in the frame at the specified position
        
        Args:
            frame: Input video frame
            ad_image: Advertisement image (with alpha channel)
            position: (x, y) coordinates for placement
            scale: Scale factor for the ad image
            
        Returns:
            Frame with ad placed
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
    
    def process_video(
        self, 
        frames: List[np.ndarray], 
        ad_image: Union[np.ndarray, Image.Image], 
        output_path: Optional[str] = None, 
        display_results: bool = True
    ) -> List[np.ndarray]:
        """
        Process video frames with ad placement
        
        Args:
            frames: List of video frames
            ad_image: Advertisement image
            output_path: Path to save the output video
            display_results: Whether to display results
            
        Returns:
            List of processed frames
        """
        from tqdm.auto import tqdm
        
        results = []
        
        # Reset state variables
        self.prev_frame = None
        self.prev_fgr = None
        self.prev_pha = None
        self.prev_state = None
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Every 10 frames, re-detect people and ad placement
            if i % 10 == 0:
                mask = self.generate_matte(frame)
                ad_position = self.find_ad_placement(frame, mask)
            
            # Place the ad
            result_frame = self.place_ad(frame, ad_image, ad_position)
            results.append(result_frame)
            
        # Save video if output path is provided
        if output_path:
            height, width = results[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
            
            for frame in results:
                out.write(frame)
                
            out.release()
            print(f"Video saved to {output_path}")
            
        return results
