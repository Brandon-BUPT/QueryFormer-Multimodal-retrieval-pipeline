# src/encoding/image_encoder.py
import torch
from PIL import Image

def encode_image(model, image: Image.Image):
    with torch.no_grad():
        feat = model.encode(images=image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu()