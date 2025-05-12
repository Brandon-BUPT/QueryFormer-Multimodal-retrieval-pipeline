# src/indexing/faiss_lsh.py
import faiss
import torch
import numpy as np
from typing import Dict, Any

class FaissLSH:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dim = config.get("dim", 512)
        self.nbits = config.get("nbits", 256)
        self.use_gpu = config.get("use_gpu", False)
    
    def create_index(self, features: torch.Tensor):
        """Create a FAISS LSH index from features."""
        features_np = features.detach().numpy().astype(np.float32)
        
        if self.use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexLSH(self.dim, self.nbits)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = faiss.IndexLSH(self.dim, self.nbits)
        
        # 添加特征向量到索引
        index.add(features_np)
        return index

# For backward compatibility
def build_faiss_lsh(features: torch.Tensor, dim: int, nbits: int, use_gpu: bool):
    """Legacy function for backward compatibility."""
    indexer = FaissLSH({"dim": dim, "nbits": nbits, "use_gpu": use_gpu})
    return indexer.create_index(features)