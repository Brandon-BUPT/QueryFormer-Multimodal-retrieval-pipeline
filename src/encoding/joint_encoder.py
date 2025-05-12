# src/encoding/joint_encoder.py
import numpy as np
from .image_encoder import encode_image
from .text_encoder import encode_text

def encode_image_text(model, tokenizer, image, text, max_token_length, stride):
    image_feat = encode_image(model, image)
    text_feat = encode_text(model, tokenizer, text, max_token_length, stride)
    joint_feat = (image_feat + text_feat) / 2
    joint_feat = joint_feat / joint_feat.norm(dim=-1, keepdim=True)
    return joint_feat.numpy().astype(np.float32)