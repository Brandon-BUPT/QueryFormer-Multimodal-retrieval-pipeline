# src/base.py
import torch
from transformers import AutoModel, AutoTokenizer
import yaml

class BaseModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_path = self.config['model']['path']
        self.device = self.config['model']['device'] if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.set_processor(self.model_path)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)