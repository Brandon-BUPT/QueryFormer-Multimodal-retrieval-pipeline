# src/encoding/joint_encoder.py
import numpy as np
import torch
from typing import Dict, Any
from PIL import Image

from .image_encoder import encode_image
from .text_encoder import encode_text

class JointEncoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.combine_method = config.get("combine_method", "average")
    
    def encode(self, model, tokenizer, image: Image.Image, text: str, max_token_length: int, stride: int):
        """Encode both image and text to create a joint representation."""
        print(f"Joint encoding image and text using '{self.combine_method}' method...")
        image_feat = encode_image(model, image)
        text_feat = encode_text(model, tokenizer, text, max_token_length, stride)
        
        # Combine features based on the specified method
        if self.combine_method == "average":
            joint_feat = (image_feat + text_feat) / 2
        elif self.combine_method == "concat":
            # In this case we would need to project back to the original dimension
            # For simplicity, we'll still use average if the method is concat
            print("Warning: Concat method not fully implemented, using average instead.")
            joint_feat = (image_feat + text_feat) / 2
        else:
            print(f"Warning: Unknown combine method '{self.combine_method}', falling back to average.")
            joint_feat = (image_feat + text_feat) / 2
            
        # Normalize the joint feature
        joint_feat = joint_feat / joint_feat.norm(dim=-1, keepdim=True)
        
        return joint_feat

# Legacy function for backward compatibility
def encode_image_text(model, tokenizer, image, text, max_token_length, stride):
    """Legacy function for backward compatibility."""
    encoder = JointEncoder({"combine_method": "average"})
    joint_feat = encoder.encode(model, tokenizer, image, text, max_token_length, stride)
    return joint_feat.numpy().astype(np.float32)