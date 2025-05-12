# src/data_preprocessing/preprocessor.py
import os
import torch
import json
import pickle
from PIL import Image
from tqdm import tqdm
from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.indexing.faiss_lsh import build_faiss_lsh

def preprocess_data(image_folder, text_jsonl, cache_dir, model, tokenizer, max_token_length, stride, faiss_dim, faiss_nbits, use_gpu):
    os.makedirs(cache_dir, exist_ok=True)
    image_feat_path = os.path.join(cache_dir, "image_features.pt")
    text_feat_path = os.path.join(cache_dir, "text_features.pt")
    meta_path = os.path.join(cache_dir, "meta.pkl")

    if os.path.exists(image_feat_path) and os.path.exists(text_feat_path) and os.path.exists(meta_path):
        print("🔁 加载本地缓存...")
        image_features = torch.load(image_feat_path)
        text_features = torch.load(text_feat_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return image_features, meta["image_paths"], text_features, meta["text_contents"], meta["text_ids"], build_faiss_lsh(image_features, faiss_dim, faiss_nbits, use_gpu), build_faiss_lsh(text_features, faiss_dim, faiss_nbits, use_gpu)

    print("📦 正在编码图像和文本特征...")

    image_features = []
    image_paths = []
    for fname in tqdm(os.listdir(image_folder), desc="Encoding Images"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, fname)
            try:
                image = Image.open(path).convert("RGB")
                image_feat = encode_image(model, image)
                image_features.append(image_feat)
                image_paths.append(path)
            except Exception as e:
                print(f"跳过图像 {path}: {e}")

    image_features = torch.cat(image_features, dim=0)

    text_features = []
    text_contents = []
    text_ids = []
    with open(text_jsonl, 'r') as f:
        for line in tqdm(f, desc="Encoding Texts"):
            obj = json.loads(line)
            text_feat = encode_text(model, tokenizer, obj["contents"], max_token_length, stride)
            text_features.append(text_feat)
            text_contents.append(obj["contents"])
            text_ids.append(obj["id"])

    text_features = torch.cat(text_features, dim=0)

    torch.save(image_features, image_feat_path)
    torch.save(text_features, text_feat_path)
    with open(meta_path, "wb") as f:
        pickle.dump({
            "image_paths": image_paths,
            "text_contents": text_contents,
            "text_ids": text_ids
        }, f)

    return image_features, image_paths, text_features, text_contents, text_ids, build_faiss_lsh(image_features, faiss_dim, faiss_nbits, use_gpu), build_faiss_lsh(text_features, faiss_dim, faiss_nbits, use_gpu)