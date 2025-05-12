#!/usr/bin/env python
# run_pipeline.py - Pipeline使用示例脚本

import os
import argparse
import yaml
from PIL import Image
from pipelines import PipelineRegistry

def parse_args():
    parser = argparse.ArgumentParser(description="多模态检索系统 - Pipeline使用示例")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml", 
                        help="Pipeline配置文件路径")
    parser.add_argument("--mode", type=str, choices=["text2image", "image2text", "multimodal2text", "text2text"],
                        default="text2image", help="查询模式")
    parser.add_argument("--text", type=str, help="文本查询")
    parser.add_argument("--image", type=str, help="图像查询路径")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载pipeline配置
    config_path = os.path.abspath(args.config)
    print(f"加载pipeline配置： {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 更新top_k参数
    config["top_k"] = args.top_k
    
    # 创建pipeline
    print(f"创建{args.mode}检索pipeline...")
    pipeline = PipelineRegistry.get_pipeline("retrieval", config)
    
    # 根据模式执行查询
    if args.mode == "text2image":
        if not args.text:
            raise ValueError("文本查询模式下，--text参数是必须的")
        
        print(f"使用文本 '{args.text}' 检索图像...")
        results = pipeline.run({"query_type": "text2image", "text": args.text})
        
        print(f"\n检索到{len(results)}个匹配图像:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['path']} (相似度: {result['similarity']:.4f})")
    
    elif args.mode == "text2text":
        if not args.text:
            raise ValueError("文本查询模式下，--text参数是必须的")
        
        print(f"使用文本 '{args.text}' 检索相关文本...")
        results = pipeline.run({"query_type": "text2text", "text": args.text})
        
        print(f"\n检索到{len(results)}个匹配文本:")
        for i, result in enumerate(results):
            print(f"{i+1}. [{result['id']}] (相似度: {result['similarity']:.4f})")
            print(f"   {result['content'][:100]}...")
            print()
    
    elif args.mode == "image2text":
        if not args.image:
            raise ValueError("图像查询模式下，--image参数是必须的")
        
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"图像文件未找到: {args.image}")
        
        print(f"使用图像 '{args.image}' 检索文本...")
        image = Image.open(args.image).convert("RGB")
        results = pipeline.run({"query_type": "image2text", "image": image})
        
        print(f"\n检索到{len(results)}个匹配文本:")
        for i, result in enumerate(results):
            print(f"{i+1}. [{result['id']}] (相似度: {result['similarity']:.4f})")
            print(f"   {result['content'][:100]}...")
            print()
    
    elif args.mode == "multimodal2text":
        if not args.image or not args.text:
            raise ValueError("多模态查询模式下，--image和--text参数都是必须的")
        
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"图像文件未找到: {args.image}")
        
        print(f"使用图像 '{args.image}' 和文本 '{args.text}' 进行多模态检索...")
        image = Image.open(args.image).convert("RGB")
        results = pipeline.run({
            "query_type": "multimodal2text", 
            "image": image,
            "text": args.text
        })
        
        print(f"\n检索到{len(results)}个匹配文本:")
        for i, result in enumerate(results):
            print(f"{i+1}. [{result['id']}] (相似度: {result['similarity']:.4f})")
            print(f"   {result['content'][:100]}...")
            print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        traceback.print_exc() 