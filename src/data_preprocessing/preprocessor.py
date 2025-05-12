# src/data_preprocessing/preprocessor.py
import os
import torch
import json
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.indexing.faiss_lsh import build_faiss_lsh


class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = config.get("cache_dir", "cache")
        self.max_token_length = config.get("max_token_length", 512)
        self.stride = config.get("stride", 256)

    def process_data(self, data_config: Dict[str, Any], model, tokenizer, indexer_factory) -> Dict[str, Any]:
        """Process data and return features, paths, indices, etc."""
        image_folder = data_config.get("image_folder", "data/images")
        text_jsonl = data_config.get("text_jsonl", "data/texts.jsonl")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        image_feat_path = os.path.join(self.cache_dir, "image_features.pt")
        text_feat_path = os.path.join(self.cache_dir, "text_features.pt")
        meta_path = os.path.join(self.cache_dir, "meta.pkl")

        if os.path.exists(image_feat_path) and os.path.exists(text_feat_path) and os.path.exists(meta_path):
            print(f"🔁 Loading from local cache at {self.cache_dir}...")
            image_features = torch.load(image_feat_path)
            text_features = torch.load(text_feat_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                
            # Create indices
            print("Building FAISS indices from cached features...")
            image_index = indexer_factory.create_index(image_features)
            text_index = indexer_factory.create_index(text_features)
            
            return {
                "image_features": image_features,
                "image_paths": meta["image_paths"],
                "text_features": text_features,
                "text_contents": meta["text_contents"],
                "text_ids": meta["text_ids"],
                "image_index": image_index,
                "text_index": text_index
            }

        print(f"📦 Encoding image and text features, will cache to {self.cache_dir}...")
        image_features, image_paths = self._process_images(image_folder, model)
        text_features, text_contents, text_ids = self._process_texts(text_jsonl, model, tokenizer)

        # Save to cache
        os.makedirs(os.path.dirname(image_feat_path), exist_ok=True)
        torch.save(image_features, image_feat_path)
        torch.save(text_features, text_feat_path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "image_paths": image_paths,
                "text_contents": text_contents,
                "text_ids": text_ids
            }, f)

        # Create indices
        print("Building FAISS indices from new features...")
        image_index = indexer_factory.create_index(image_features)
        text_index = indexer_factory.create_index(text_features)
        
        return {
            "image_features": image_features,
            "image_paths": image_paths,
            "text_features": text_features,
            "text_contents": text_contents,
            "text_ids": text_ids,
            "image_index": image_index,
            "text_index": text_index
        }

    def _process_images(self, image_folder: str, model) -> Tuple[torch.Tensor, List[str]]:
        """Process images and return features and paths."""
        image_features = []
        image_paths = []
        
        print(f"Processing images from: {image_folder}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
            
        for fname in tqdm(os.listdir(image_folder), desc="Encoding Images"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(image_folder, fname)
                try:
                    image = Image.open(path).convert("RGB")
                    image_feat = encode_image(model, image)
                    image_features.append(image_feat)
                    image_paths.append(path)
                except Exception as e:
                    print(f"Skipping image {path}: {e}")

        if not image_features:
            raise ValueError(f"No valid images found in {image_folder}")
            
        return torch.cat(image_features, dim=0), image_paths

    def _process_texts(self, text_jsonl: str, model, tokenizer) -> Tuple[torch.Tensor, List[str], List[str]]:
        """Process texts and return features, contents, and IDs."""
        text_features = []
        text_contents = []
        text_ids = []
        
        print(f"Processing texts from: {text_jsonl}")
        if not os.path.exists(text_jsonl):
            raise FileNotFoundError(f"Text JSONL file not found: {text_jsonl}")
            
        with open(text_jsonl, 'r') as f:
            for line in tqdm(f, desc="Encoding Texts"):
                obj = json.loads(line)
                text_feat = encode_text(model, tokenizer, obj["contents"], self.max_token_length, self.stride)
                text_features.append(text_feat)
                text_contents.append(obj["contents"])
                text_ids.append(obj["id"])

        if not text_features:
            raise ValueError(f"No valid texts found in {text_jsonl}")
            
        return torch.cat(text_features, dim=0), text_contents, text_ids