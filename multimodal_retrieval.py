#!/usr/bin/env python
# multimodal_retrieval.py

import os
import gradio as gr
import yaml
from pipelines import PipelineRegistry

def main():
    # 确保配置文件路径正确
    retrieval_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "config/pipeline_config.yaml")
    query_analysis_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "config/query_analysis_config.yaml")
    
    print(f"Loading pipeline configuration from {retrieval_config_path}")
    # 加载pipeline配置
    with open(retrieval_config_path, "r") as f:
        retrieval_config = yaml.safe_load(f)
    
    with open(query_analysis_config_path, "r") as f:
        query_analysis_config = yaml.safe_load(f)
    
    # 从注册表创建检索pipeline
    retrieval_pipeline = PipelineRegistry.get_pipeline("retrieval", retrieval_config)
    query_analysis_pipeline = PipelineRegistry.get_pipeline("query_analysis", query_analysis_config)
    
    # Gradio UI
    with gr.Blocks(title="BGE-VL Multimodal Retrieval System") as demo:
        gr.Markdown("## 🔍 多模态检索系统 (FAISS LSH)")

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                txt = gr.Textbox(label="输入文本查询")
                btn = gr.Button("搜索图像")
                gallery = gr.Gallery(label="匹配的图像", columns=3)
                
                def text2image_search(query):
                    if not query or query.strip() == "":
                        return []
                    results = retrieval_pipeline.run({"query_type": "text2image", "text": query})
                    return [item["path"] for item in results]
                
                btn.click(fn=text2image_search, inputs=txt, outputs=gallery)

            with gr.Tab("Text → Text"):
                txt_query = gr.Textbox(label="输入文本查询")
                btn_txt = gr.Button("搜索文本")
                out_text_result = gr.Markdown(label="匹配的文本")
                
                def text2text_search(query):
                    if not query or query.strip() == "":
                        return "请输入查询文本。"
                    
                    results = retrieval_pipeline.run({"query_type": "text2text", "text": query})
                    if not results:
                        return "未找到匹配的文本。"
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn_txt.click(fn=text2text_search, inputs=txt_query, outputs=out_text_result)

            with gr.Tab("Image → Text"):
                img = gr.Image(label="上传图像", type="pil")
                btn2 = gr.Button("搜索文本")
                out_text = gr.Markdown(label="匹配的文本")
                
                def image2text_search(image):
                    if image is None:
                        return "请上传图像。"
                    
                    results = retrieval_pipeline.run({"query_type": "image2text", "image": image})
                    if not results:
                        return "未找到匹配的文本。"
                    return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results])
                
                btn2.click(fn=image2text_search, inputs=img, outputs=out_text)

            with gr.Tab("Image + Text → Text"):
                with gr.Row():
                    with gr.Column():
                        img2 = gr.Image(label="上传图像", type="pil")
                        txt2 = gr.Textbox(label="输入文本查询")
                        use_analysis = gr.Checkbox(label="使用查询分析", value=True)
                        btn3 = gr.Button("搜索文本")
                    
                    with gr.Column():
                        analysis_info = gr.Markdown(label="查询分析结果", value="")
                
                out_text2 = gr.Markdown(label="匹配的文本")
                
                def multimodal2text_search(image, text, use_query_analysis):
                    if image is None:
                        return "请上传图像。", "未进行查询分析"
                    if not text or text.strip() == "":
                        return "请输入查询文本。", "未进行查询分析"
                    
                    if use_query_analysis:
                        # 使用查询分析管道
                        results = query_analysis_pipeline.run({
                            "query_type": "multimodal2text", 
                            "image": image, 
                            "text": text
                        })
                        
                        # 提取分析信息
                        analysis_markdown = ""
                        if results.get("enhanced_query"):
                            analysis_markdown += f"**原始查询**: {results['original_query']}\n\n"
                            analysis_markdown += f"**增强查询**: {results['enhanced_query']}\n\n"
                            analysis_markdown += "**关键词**:\n"
                            analysis_markdown += f"- 显式关键词: {', '.join(results['keywords']['explicit'])}\n"
                            analysis_markdown += f"- 隐式关键词: {', '.join(results['keywords']['implicit'])}\n"
                        else:
                            analysis_markdown = "查询分析失败，使用原始查询。"
                        
                        # 返回检索结果和分析信息
                        retrieval_results = results["results"]
                        if not retrieval_results:
                            return "未找到匹配的文本。", analysis_markdown
                        return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in retrieval_results]), analysis_markdown
                    else:
                        # 使用标准检索管道
                        results = retrieval_pipeline.run({
                            "query_type": "multimodal2text", 
                            "image": image, 
                            "text": text
                        })
                        if not results:
                            return "未找到匹配的文本。", "未使用查询分析"
                        return "\n\n".join([f"**[{item['id']}]**  \n{item['content']}" for item in results]), "未使用查询分析"
                
                btn3.click(fn=multimodal2text_search, inputs=[img2, txt2, use_analysis], outputs=[out_text2, analysis_info])

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