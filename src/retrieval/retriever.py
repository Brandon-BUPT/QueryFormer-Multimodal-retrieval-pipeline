# src/retrieval/retriever.py
import numpy as np
from PIL import Image
from typing import Dict, Any, List

from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.encoding.joint_encoder import encode_image_text

class Retriever:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.top_k = config.get("top_k", 3)
        print(f"Retriever initialized with top_k={self.top_k}")
    
    def retrieve_by_text(self, query: str, model, tokenizer, max_token_length: int, stride: int, image_index, image_paths):
        """Retrieve images based on text query."""
        print(f"Retrieving top {self.top_k} images for text query")
        query_features = encode_text(model, tokenizer, query, max_token_length, stride)
        D, I = image_index.search(query_features.numpy().astype(np.float32), self.top_k)
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            if i < len(image_paths):  # Ensure index is valid
                results.append({
                    "path": image_paths[i],
                    "similarity": float(similarity),
                    "type": "image"
                })
            else:
                print(f"Warning: Invalid index {i} for image_paths with length {len(image_paths)}")
        
        print(f"Found {len(results)} matching images")
        return results
    
    def retrieve_by_image(self, image: Image.Image, model, text_index, text_ids, text_contents):
        """Retrieve texts based on image query."""
        print(f"Retrieving top {self.top_k} texts for image query")
        query_features = encode_image(model, image)
        D, I = text_index.search(query_features.numpy().astype(np.float32), self.top_k)
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            if i < len(text_ids):  # Ensure index is valid
                results.append({
                    "id": text_ids[i],
                    "content": text_contents[i],
                    "similarity": float(similarity),
                    "type": "text"
                })
            else:
                print(f"Warning: Invalid index {i} for text_ids with length {len(text_ids)}")
        
        print(f"Found {len(results)} matching texts")
        return results
    
    def retrieve_by_image_and_text(self, image: Image.Image, text: str, model, tokenizer, 
                                  max_token_length: int, stride: int, text_index, text_ids, text_contents):
        """Retrieve texts based on combined image and text query."""
        print(f"Retrieving top {self.top_k} texts for multimodal query")
        query_feat = encode_image_text(model, tokenizer, image, text, max_token_length, stride)
        D, I = text_index.search(query_feat, self.top_k)
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            if i < len(text_ids):  # Ensure index is valid
                results.append({
                    "id": text_ids[i],
                    "content": text_contents[i],
                    "similarity": float(similarity),
                    "type": "text"
                })
            else:
                print(f"Warning: Invalid index {i} for text_ids with length {len(text_ids)}")
        
        print(f"Found {len(results)} matching texts")
        return results


# Legacy functions for backward compatibility
def retrieve_by_text(query, image_index, image_paths, model, tokenizer, max_token_length, stride, top_k=3):
    """Legacy function for backward compatibility."""
    retriever = Retriever({"top_k": top_k})
    results = retriever.retrieve_by_text(query, model, tokenizer, max_token_length, stride, image_index, image_paths)
    return [result["path"] for result in results]

def retrieve_by_image(uploaded_img, text_index, text_ids, text_contents, model, top_k=3):
    """Legacy function for backward compatibility."""
    retriever = Retriever({"top_k": top_k})
    results = retriever.retrieve_by_image(uploaded_img, model, text_index, text_ids, text_contents)
    return "\n\n".join([f"**[{result['id']}]**  \n{result['content']}" for result in results])

def retrieve_by_image_and_text(image, text, text_index, text_ids, text_contents, model, tokenizer, max_token_length, stride, top_k=3):
    """Legacy function for backward compatibility."""
    retriever = Retriever({"top_k": top_k})
    results = retriever.retrieve_by_image_and_text(image, text, model, tokenizer, max_token_length, stride, text_index, text_ids, text_contents)
    return "\n\n".join([f"**[{result['id']}]**  \n{result['content']}" for result in results])