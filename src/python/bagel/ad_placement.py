"""
BAGELAdPlacement - Intelligent ad placement using BAGEL model
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple

class BAGELAdPlacement:
    def __init__(self, bagel_inferencer):
        """
        Initialize the BAGEL ad placement module
        
        Args:
            bagel_inferencer: The BAGEL model inferencer
        """
        self.bagel_inferencer = bagel_inferencer
        
    def find_optimal_placement(self, frame, mask, scene_info=None) -> Optional[Tuple[int, int]]:
        """
        Find optimal ad placement using BAGEL's scene understanding
        
        Args:
            frame: The video frame
            mask: Segmentation mask
            scene_info: Scene analysis information
            
        Returns:
            Tuple of (x, y) coordinates for ad placement, or None if no suitable placement found
        """
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return None
        if not contours:
            return None
        
        # Find the largest contour (assuming it's a person)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 1000:  # Minimum size threshold
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Use BAGEL for intelligent placement if available
        if self.bagel_inferencer is not None and scene_info is not None:
            try:
                # Convert frame to PIL Image for BAGEL
                if isinstance(frame, np.ndarray):
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    pil_frame = frame
                    
                # Generate a prompt for BAGEL to analyze optimal ad placement
                context_str = ", ".join(scene_info["context"]) if "context" in scene_info else "general"
                prompt = f"Where would be the best place to put an advertisement in this {context_str} scene? Answer with one of: right, left, top, bottom."
                
                # Use BAGEL to analyze the frame for optimal placement
                output_dict = self.bagel_inferencer(
                    image=pil_frame,
                    text=prompt,
                    cfg_text_scale=3.0,
                    num_timesteps=20,
                )
                
                placement_analysis = output_dict["text"]
                
                # Parse the placement recommendation
                if "right" in placement_analysis.lower():
                    ad_x = x + w + 20  # Place to the right
                    ad_y = y
                elif "left" in placement_analysis.lower():
                    ad_x = max(0, x - 100)  # Place to the left
                    ad_y = y
                elif "top" in placement_analysis.lower():
                    ad_x = x
                    ad_y = max(0, y - 100)  # Place above
                elif "bottom" in placement_analysis.lower():
                    ad_x = x
                    ad_y = y + h + 20  # Place below
                else:
                    # Default to right side placement
                    ad_x = x + w + 20
                    ad_y = y
                    
                return (ad_x, ad_y)
                
            except Exception as e:
                print(f"Error in BAGEL ad placement: {e}")
                # Fall back to default placement
        
        # Default placement (to the right of the person)
        ad_x = x + w + 20
        ad_y = y
        
        return (ad_x, ad_y)
