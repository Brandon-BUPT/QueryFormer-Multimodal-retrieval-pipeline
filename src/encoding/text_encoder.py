# src/encoding/text_encoder.py
import torch

def encode_text(model, tokenizer, text: str, max_token_length: int, stride: int):
    tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens['input_ids'][0]

    if len(input_ids) <= max_token_length:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_token_length)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        with torch.no_grad():
            feat = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu()

    segment_feats = []
    for start in range(0, len(input_ids), stride):
        end = start + max_token_length
        chunk_ids = input_ids[start:end]
        if len(chunk_ids) == 0:
            continue
        chunk_ids = chunk_ids.unsqueeze(0).to(model.device)
        attention_mask = torch.ones_like(chunk_ids).to(model.device)
        with torch.no_grad():
            feat = model.get_text_features(input_ids=chunk_ids, attention_mask=attention_mask)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            segment_feats.append(feat.cpu())

    final_feat = torch.mean(torch.cat(segment_feats, dim=0), dim=0, keepdim=True)
    final_feat = final_feat / final_feat.norm(dim=-1, keepdim=True)
    return final_feat