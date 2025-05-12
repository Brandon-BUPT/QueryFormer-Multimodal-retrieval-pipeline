# examples/multimodal_retrieval.py

import gradio as gr
from src.base import BaseModel
from src.data_preprocessing.preprocessor import preprocess_data
from src.retrieval.retriever import retrieve_by_text, retrieve_by_image, retrieve_by_image_and_text

# 加载配置和模型
model = BaseModel("config/model_config.yaml")
config = model.config
image_folder = config['data']['image_folder']
text_jsonl = config['data']['text_jsonl']
cache_dir = config['model']['cache_dir']
max_token_length = config['model']['max_token_length']
stride = config['model']['stride']
faiss_dim = config['model']['faiss']['dim']
faiss_nbits = config['model']['faiss']['nbits']
use_gpu = config['model']['faiss']['use_gpu']

# 预处理数据
image_features, image_paths, text_features, text_contents, text_ids, image_index, text_index = preprocess_data(
    image_folder, text_jsonl, cache_dir, model.model, model.tokenizer, max_token_length, stride, faiss_dim, faiss_nbits, use_gpu
)

# Gradio UI
with gr.Blocks(title="BGE-VL Multimodal Retrieval System") as demo:
    gr.Markdown("## 🔍 Multimodal Retrieval System (FAISS LSH)")

    with gr.Tabs():
        with gr.Tab("Text → Image"):
            txt = gr.Textbox(label="Enter your text query")
            btn = gr.Button("Search Images")
            gallery = gr.Gallery(label="Top Matching Images", columns=3)
            btn.click(
                fn=lambda query: retrieve_by_text(query, image_index, image_paths, model.model, model.tokenizer, max_token_length, stride),
                inputs=txt, outputs=gallery
            )

        with gr.Tab("Image → Text"):
            img = gr.Image(label="Upload Image", type="pil")
            btn2 = gr.Button("Search Texts")
            out_text = gr.Markdown(label="Top Matching Texts")
            btn2.click(
                fn=lambda uploaded_img: retrieve_by_image(uploaded_img, text_index, text_ids, text_contents, model.model),
                inputs=img, outputs=out_text
            )

        with gr.Tab("Image + Text → Text"):
            img2 = gr.Image(label="Upload Image", type="pil")
            txt2 = gr.Textbox(label="Enter accompanying text")
            btn3 = gr.Button("Search Texts")
            out_text2 = gr.Markdown(label="Top Matching Texts")
            btn3.click(
                fn=lambda image, text: retrieve_by_image_and_text(image, text, text_index, text_ids, text_contents, model.model, model.tokenizer, max_token_length, stride),
                inputs=[img2, txt2], outputs=out_text2
            )

if __name__ == '__main__':
    demo.launch()