"""
BAGELSceneUnderstanding - Scene analysis using BAGEL model
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any, Optional

class BAGELSceneUnderstanding:
    def __init__(self, bagel_inferencer):
        """
        Initialize the BAGEL scene understanding module
        
        Args:
            bagel_inferencer: The BAGEL model inferencer
        """
        self.bagel_inferencer = bagel_inferencer
        self.scene_cache = {}  # Cache scene analysis to avoid redundant processing
        
    def analyze_frame(self, frame, frame_idx):
        """
        Analyze frame using BAGEL to extract scene context and objects
        
        Args:
            frame: The video frame to analyze
            frame_idx: Frame index for caching purposes
            
        Returns:
            Dict containing scene analysis information
        """
        # Check cache first to avoid redundant processing
        if frame_idx in self.scene_cache:
            return self.scene_cache[frame_idx]
            
        # Convert frame to PIL Image for BAGEL
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_frame = frame
            
        # Use BAGEL for scene understanding if available
        if self.bagel_inferencer is not None:
            try:
                # Use BAGEL for scene understanding
                prompt = "Describe this scene in detail, focusing on the environment, people, objects, and overall context."
                
                # Call the BAGEL inferencer with the image and prompt
                output_dict = self.bagel_inferencer(
                    image=pil_frame,
                    text=prompt,
                    think=True,  # Enable thinking mode for detailed analysis
                    cfg_text_scale=4.0,  # Control text guidance
                    num_timesteps=20,    # Faster inference for analysis
                )
                
                scene_analysis = output_dict["text"]
                
                # Extract key information from BAGEL's analysis
                scene_info = {
                    "description": scene_analysis,
                    "context": self._extract_context(scene_analysis),
                    "objects": self._extract_objects(scene_analysis),
                    "mood": self._extract_mood(scene_analysis),
                    "suitable_ad_types": self._recommend_ad_types(scene_analysis)
                }
            except Exception as e:
                print(f"Error in BAGEL scene analysis: {e}")
                scene_info = self._fallback_analysis(frame)
        else:
            # Fallback to basic analysis if BAGEL is not available
            scene_info = self._fallback_analysis(frame)
            
        # Cache the results
        self.scene_cache[frame_idx] = scene_info
        return scene_info
    
    def _extract_context(self, analysis):
        """
        Extract context keywords from the scene analysis
        
        Args:
            analysis: Text analysis from BAGEL
            
        Returns:
            List of context keywords
        """
        # Simple keyword extraction - in a real system this would be more sophisticated
        keywords = []
        if "outdoor" in analysis.lower():
            keywords.append("outdoor")
        if "indoor" in analysis.lower():
            keywords.append("indoor")
        if "crowd" in analysis.lower():
            keywords.append("crowded")
        if "urban" in analysis.lower():
            keywords.append("urban")
        if "rural" in analysis.lower():
            keywords.append("rural")
        if "sports" in analysis.lower() or "game" in analysis.lower():
            keywords.append("sports")
        if "concert" in analysis.lower() or "music" in analysis.lower():
            keywords.append("entertainment")
        
        return keywords if keywords else ["general"]
    
    def _extract_objects(self, analysis):
        """
        Extract objects mentioned in the scene analysis
        
        Args:
            analysis: Text analysis from BAGEL
            
        Returns:
            List of detected objects
        """
        objects = []
        common_objects = ["person", "people", "car", "building", "tree", "sign", "phone", "computer"]
        
        for obj in common_objects:
            if obj in analysis.lower():
                objects.append(obj)
                
        return objects
    
    def _extract_mood(self, analysis):
        """
        Extract mood/atmosphere from the scene analysis
        
        Args:
            analysis: Text analysis from BAGEL
            
        Returns:
            List of detected moods
        """
        moods = []
        mood_keywords = {
            "happy": ["happy", "cheerful", "joyful", "excited"],
            "serious": ["serious", "formal", "professional"],
            "relaxed": ["relaxed", "calm", "peaceful"],
            "energetic": ["energetic", "dynamic", "active"],
            "tense": ["tense", "anxious", "stressed"]
        }
        
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword in analysis.lower():
                    moods.append(mood)
                    break
                    
        return moods if moods else ["neutral"]
    
    def _recommend_ad_types(self, analysis):
        """
        Recommend suitable ad types based on scene analysis
        
        Args:
            analysis: Text analysis from BAGEL
            
        Returns:
            List of recommended ad types
        """
        ad_types = []
        
        # Context-based recommendations
        if any(kw in analysis.lower() for kw in ["sports", "game", "athletic", "fitness"]):
            ad_types.append("sports_equipment")
            ad_types.append("energy_drinks")
            
        if any(kw in analysis.lower() for kw in ["food", "restaurant", "eating", "dining"]):
            ad_types.append("food_delivery")
            ad_types.append("restaurants")
            
        if any(kw in analysis.lower() for kw in ["outdoor", "nature", "park"]):
            ad_types.append("outdoor_gear")
            ad_types.append("travel")
            
        if any(kw in analysis.lower() for kw in ["technology", "phone", "computer", "digital"]):
            ad_types.append("electronics")
            ad_types.append("software")
            
        # Default to general ads if no specific context is detected
        if not ad_types:
            ad_types = ["general_retail", "services"]
            
        return ad_types
    
    def _fallback_analysis(self, frame):
        """
        Perform basic analysis when BAGEL is not available
        
        Args:
            frame: The video frame
            
        Returns:
            Dict with basic scene information
        """
        # Simple fallback that provides basic information
        return {
            "description": "A scene with potential crowd elements.",
            "context": ["general"],
            "objects": ["person"],
            "mood": ["neutral"],
            "suitable_ad_types": ["general_retail", "services"]
        }
