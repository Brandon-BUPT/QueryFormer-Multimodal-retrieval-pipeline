# src/indexing/faiss_lsh.py
import faiss
import torch
import numpy as np

def build_faiss_lsh(features: torch.Tensor, dim: int, nbits: int, use_gpu: bool):
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexLSH(dim, nbits)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = faiss.IndexLSH(dim, nbits)
    index.add(features.detach().numpy().astype(np.float32))
    return index