"""
BAGELAdOptimization - Context-aware ad content optimization using BAGEL model
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any, Optional, Union

class BAGELAdOptimization:
    def __init__(self, bagel_inferencer):
        """
        Initialize the BAGEL ad optimization module
        
        Args:
            bagel_inferencer: The BAGEL model inferencer
        """
        self.bagel_inferencer = bagel_inferencer
        self.ad_cache = {}  # Cache generated ads
        
    def optimize_ad_for_scene(self, ad_image, scene_info, ad_type=None) -> Union[np.ndarray, Image.Image]:
        """
        Optimize ad content for the specific scene context
        
        Args:
            ad_image: The original advertisement image
            scene_info: Scene analysis information
            ad_type: Type of advertisement
            
        Returns:
            Optimized advertisement image
        """
        # If BAGEL is not available, return original ad
        if self.bagel_inferencer is None:
            return ad_image
            
        # Generate cache key
        context_str = ",".join(scene_info["context"]) if "context" in scene_info else "general"
        mood_str = ",".join(scene_info["mood"]) if "mood" in scene_info else "neutral"
        cache_key = f"{ad_type}_{context_str}_{mood_str}"
        
        # Check cache first
        if cache_key in self.ad_cache:
            return self.ad_cache[cache_key]
            
        try:
            # Convert ad_image to PIL Image if it's not already
            if isinstance(ad_image, np.ndarray):
                pil_ad = Image.fromarray(cv2.cvtColor(ad_image, cv2.COLOR_BGR2RGB))
            else:
                pil_ad = ad_image
                
            # Create a prompt for BAGEL to optimize the ad
            prompt = f"Modify this advertisement to better match a {mood_str} scene with {context_str} context."
            
            # Use BAGEL to generate an optimized ad
            output_dict = self.bagel_inferencer(
                image=pil_ad,
                text=prompt,
                cfg_text_scale=6.0,
                cfg_img_scale=1.5,
                num_timesteps=50,
            )
            
            optimized_ad = output_dict["image"]
            
            # Cache the result
            self.ad_cache[cache_key] = optimized_ad
            return optimized_ad
                
        except Exception as e:
            print(f"Error in BAGEL ad optimization: {e}")
            # Fall back to original ad
            return ad_image
