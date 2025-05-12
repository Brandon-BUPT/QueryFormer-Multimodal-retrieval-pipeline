#!/usr/bin/env python
# multimodal_retrieval.py

import os
import gradio as gr
import yaml
from pipelines import PipelineRegistry

def main():
    # 确保配置文件路径正确
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "config/pipeline_config.yaml")
    
    print(f"Loading pipeline configuration from {config_path}")
    # 加载pipeline配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 从注册表创建检索pipeline
    pipeline = PipelineRegistry.get_pipeline("retrieval", config)
    
    # Gradio UI
    with gr.Blocks(title="BGE-VL Multimodal Retrieval System") as demo:
        gr.Markdown("## 🔍 Multimodal Retrieval System (FAISS LSH)")

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                txt = gr.Textbox(label="Enter your text query")
                btn = gr.Button("Search Images")
                gallery = gr.Gallery(label="Top Matching Images", columns=3)
                
                def text2image_search(query):
                    if not query or query.strip() == "":
                        return []
                    results = pipeline.run({"query_type": "text2image", "text": query})
                    return [item["path"] for item in results]
                
                btn.click(fn=text2image_search, inputs=txt, outputs=gallery)

            with gr.Tab("Text → Text"):
                txt_query = gr.Textbox(label="Enter your text query")
                btn_txt = gr.Button("Search Texts")
                out_text_result = gr.Markdown(label="Top Matching Texts")
                
                def text2text_search(query):
                    if not query or query.strip() == "":
                        return "Please enter a text query."
                    
                    results = pipeline.run({"query_type": "text2text", "text": query})
                    if not results:
                        return "No matching texts found."
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn_txt.click(fn=text2text_search, inputs=txt_query, outputs=out_text_result)

            with gr.Tab("Image → Text"):
                img = gr.Image(label="Upload Image", type="pil")
                btn2 = gr.Button("Search Texts")
                out_text = gr.Markdown(label="Top Matching Texts")
                
                def image2text_search(image):
                    if image is None:
                        return "Please upload an image."
                    
                    results = pipeline.run({"query_type": "image2text", "image": image})
                    if not results:
                        return "No matching texts found."
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn2.click(fn=image2text_search, inputs=img, outputs=out_text)

            with gr.Tab("Image + Text → Text"):
                img2 = gr.Image(label="Upload Image", type="pil")
                txt2 = gr.Textbox(label="Enter accompanying text")
                btn3 = gr.Button("Search Texts")
                out_text2 = gr.Markdown(label="Top Matching Texts")
                
                def multimodal2text_search(image, text):
                    if image is None:
                        return "Please upload an image."
                    if not text or text.strip() == "":
                        return "Please enter some text."
                    
                    results = pipeline.run({
                        "query_type": "multimodal2text", 
                        "image": image, 
                        "text": text
                    })
                    if not results:
                        return "No matching texts found."
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn3.click(fn=multimodal2text_search, inputs=[img2, txt2], outputs=out_text2)

    # 启动Gradio界面
    print("Starting Gradio interface...")
    demo.launch(share=False)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()