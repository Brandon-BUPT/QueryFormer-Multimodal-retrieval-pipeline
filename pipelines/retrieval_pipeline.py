from typing import Dict, Any, List
import torch
import numpy as np
from PIL import Image
import os

from .base_pipeline import BasePipeline
from .factory import ComponentFactory
from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.encoding.joint_encoder import encode_image_text


class RetrievalPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.preprocessed_data = self._load_or_preprocess_data()
        print("🚀 RetrievalPipeline initialized successfully!")

    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize encoder, indexer and retriever components."""
        components = {}
        
        print("⚙️ Initializing pipeline components...")
        
        # Initialize model and tokenizer
        from src.base import BaseModel
        model_config_path = os.path.abspath(self.config["model_config_path"])
        print(f"Loading base model from: {model_config_path}")
        model = BaseModel(model_config_path)
        components["model"] = model.model
        components["tokenizer"] = model.tokenizer
        components["device"] = model.device
        
        # Initialize encoder
        components["encoder"] = ComponentFactory.create_component(
            "encoder", self.config["encoder"]
        )
        
        # Initialize indexer
        components["indexer"] = ComponentFactory.create_component(
            "indexer", self.config["indexer"]
        )
        
        # Initialize retriever
        components["retriever"] = ComponentFactory.create_component(
            "retriever", self.config["retriever"]
        )
        
        return components
    
    def _load_or_preprocess_data(self):
        """Load preprocessed data or preprocess it if needed."""
        print("🔄 Loading or preprocessing data...")
        preprocessor = ComponentFactory.create_component(
            "preprocessor", self.config["preprocessor"]
        )
        
        return preprocessor.process_data(
            self.config["data"],
            self.components["model"],
            self.components["tokenizer"],
            self.components["indexer"]
        )
    
    def run(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the retrieval pipeline based on the query type."""
        query_type = input_data["query_type"]
        print(f"🔍 Running {query_type} query...")
        
        if query_type == "text2image":
            return self._retrieve_by_text(input_data["text"])
        elif query_type == "image2text":
            return self._retrieve_by_image(input_data["image"])
        elif query_type == "multimodal2text":
            return self._retrieve_by_image_and_text(input_data["image"], input_data["text"])
        elif query_type == "text2text":
            return self._retrieve_text_by_text(input_data["text"])
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
    
    def _retrieve_by_text(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve images based on text query."""
        print(f"Processing text query: {query_text[:50]}...")
        query_features = encode_text(
            self.components["model"], 
            self.components["tokenizer"], 
            query_text, 
            self.config["max_token_length"], 
            self.config["stride"]
        )
        
        # Search in the image index
        D, I = self.preprocessed_data["image_index"].search(
            query_features.numpy().astype(np.float32), 
            self.config["top_k"]
        )
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            results.append({
                "path": self.preprocessed_data["image_paths"][i],
                "similarity": float(similarity),
                "type": "image"
            })
        
        return results
    
    def _retrieve_text_by_text(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve texts based on text query."""
        print(f"Processing text2text query: {query_text[:50]}...")
        query_features = encode_text(
            self.components["model"], 
            self.components["tokenizer"], 
            query_text, 
            self.config["max_token_length"], 
            self.config["stride"]
        )
        
        # Search in the text index
        D, I = self.preprocessed_data["text_index"].search(
            query_features.numpy().astype(np.float32), 
            self.config["top_k"]
        )
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            if i < len(self.preprocessed_data["text_ids"]):
                results.append({
                    "id": self.preprocessed_data["text_ids"][i],
                    "content": self.preprocessed_data["text_contents"][i],
                    "similarity": float(similarity),
                    "type": "text"
                })
            else:
                print(f"Warning: Invalid index {i} for text_ids with length {len(self.preprocessed_data['text_ids'])}")
        
        print(f"Found {len(results)} matching texts for text query")
        return results
    
    def _retrieve_by_image(self, query_image: Image.Image) -> List[Dict[str, Any]]:
        """Retrieve texts based on image query."""
        print("Processing image query...")
        query_features = encode_image(self.components["model"], query_image)
        
        # Search in the text index
        D, I = self.preprocessed_data["text_index"].search(
            query_features.numpy().astype(np.float32), 
            self.config["top_k"]
        )
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            results.append({
                "id": self.preprocessed_data["text_ids"][i],
                "content": self.preprocessed_data["text_contents"][i],
                "similarity": float(similarity),
                "type": "text"
            })
        
        return results
    
    def _retrieve_by_image_and_text(self, query_image: Image.Image, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve texts based on combined image and text query."""
        print(f"Processing multimodal query with text: {query_text[:50]}...")
        query_feat = encode_image_text(
            self.components["model"], 
            self.components["tokenizer"], 
            query_image, 
            query_text, 
            self.config["max_token_length"], 
            self.config["stride"]
        )
        
        # Search in the text index
        D, I = self.preprocessed_data["text_index"].search(query_feat, self.config["top_k"])
        
        results = []
        for i, similarity in zip(I[0], D[0]):
            results.append({
                "id": self.preprocessed_data["text_ids"][i],
                "content": self.preprocessed_data["text_contents"][i],
                "similarity": float(similarity),
                "type": "text"
            })
        
        return results