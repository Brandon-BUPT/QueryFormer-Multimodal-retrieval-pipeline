# src/retrieval/retriever.py
import numpy as np
from PIL import Image
from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.encoding.joint_encoder import encode_image_text

def retrieve_by_text(query, image_index, image_paths, model, tokenizer, max_token_length, stride, top_k=3):
    query_features = encode_text(model, tokenizer, query, max_token_length, stride)
    D, I = image_index.search(query_features.numpy().astype(np.float32), top_k)
    return [image_paths[i] for i in I[0]]

def retrieve_by_image(uploaded_img, text_index, text_ids, text_contents, model, top_k=3):
    query_features = encode_image(model, uploaded_img)
    D, I = text_index.search(query_features.numpy().astype(np.float32), top_k)
    return "\n\n".join([f"**[{text_ids[i]}]**  \n{text_contents[i]}" for i in I[0]])

def retrieve_by_image_and_text(image: Image.Image, text: str, text_index, text_ids, text_contents, model, tokenizer, max_token_length, stride, top_k=3):
    query_feat = encode_image_text(model, tokenizer, image, text, max_token_length, stride)
    D, I = text_index.search(query_feat, top_k)
    return "\n\n".join([f"**[{text_ids[i]}]**  \n{text_contents[i]}" for i in I[0]])