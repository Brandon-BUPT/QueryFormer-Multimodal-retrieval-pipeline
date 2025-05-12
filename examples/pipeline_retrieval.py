#!/usr/bin/env python
# examples/pipeline_retrieval.py

import argparse
import gradio as gr
from PIL import Image
import yaml
from pipelines import PipelineRegistry

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Retrieval using Pipeline Architecture")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml", help="Path to pipeline configuration")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load pipeline configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create pipeline from registry
    pipeline = PipelineRegistry.get_pipeline("retrieval", config)
    
    # Create Gradio UI
    with gr.Blocks(title="Multimodal Retrieval System") as demo:
        gr.Markdown("## 🔍 Multimodal Retrieval System")
        
        with gr.Tabs():
            with gr.Tab("Text → Image"):
                txt = gr.Textbox(label="Enter your text query")
                btn = gr.Button("Search Images")
                gallery = gr.Gallery(label="Top Matching Images", columns=3)
                
                def text2image_search(query):
                    results = pipeline.run({"query_type": "text2image", "text": query})
                    return [item["path"] for item in results]
                
                btn.click(fn=text2image_search, inputs=txt, outputs=gallery)
            
            with gr.Tab("Image → Text"):
                img = gr.Image(label="Upload Image", type="pil")
                btn2 = gr.Button("Search Texts")
                out_text = gr.Markdown(label="Top Matching Texts")
                
                def image2text_search(image):
                    if image is None:
                        return "Please upload an image."
                    
                    results = pipeline.run({"query_type": "image2text", "image": image})
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
                    
                    results = pipeline.run({
                        "query_type": "multimodal2text", 
                        "image": image, 
                        "text": text
                    })
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn3.click(fn=multimodal2text_search, inputs=[img2, txt2], outputs=out_text2)
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main() 