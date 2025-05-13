#!/usr/bin/env python
# run_query_analysis.py - 查询分析管道示例脚本

import os
import argparse
import yaml
from PIL import Image
import json
from pipelines import PipelineRegistry

def parse_args():
    parser = argparse.ArgumentParser(description="多模态查询分析和检索系统 - 示例")
    parser.add_argument("--config", type=str, default="config/query_analysis_config.yaml", 
                        help="查询分析管道配置文件路径")
    parser.add_argument("--image", type=str, required=True,
                        help="图像查询路径")
    parser.add_argument("--text", type=str, required=True,
                        help="文本查询")
    parser.add_argument("--output", type=str, default="query_analysis_result.json",
                        help="输出结果的JSON文件路径")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="返回结果数量")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载管道配置
    config_path = os.path.abspath(args.config)
    print(f"加载查询分析管道配置: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 更新top_k参数
    config["top_k"] = args.top_k
    
    # 创建查询分析管道
    print(f"创建查询分析管道...")
    pipeline = PipelineRegistry.get_pipeline("query_analysis", config)
    
    # 准备输入数据
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像文件未找到: {args.image}")
    
    print(f"使用图像 '{args.image}' 和文本 '{args.text}' 进行多模态查询分析...")
    image = Image.open(args.image).convert("RGB")
    
    # 运行查询分析管道
    results = pipeline.run({
        "query_type": "multimodal2text", 
        "image": image,
        "text": args.text
    })
    
    # 显示结果
    print("\n===== 查询分析结果 =====")
    if results["enhanced_query"]:
        print(f"原始查询: {results['original_query']}")
        print(f"增强查询: {results['enhanced_query']}")
        print("\n关键词:")
        print(f"  显式关键词: {', '.join(results['keywords']['explicit'])}")
        print(f"  隐式关键词: {', '.join(results['keywords']['implicit'])}")
    else:
        print("查询分析失败，使用原始查询。")
    
    print("\n===== 检索结果 =====")
    retrieval_results = results["results"]
    for i, result in enumerate(retrieval_results):
        print(f"{i+1}. [{result['id']}] (相似度: {result['similarity']:.4f})")
        # 只显示内容的前100个字符
        content = result['content']
        print(f"   {content[:100]}...")
        if len(content) > 100:
            print("   ...")
        print()
    
    # 保存结果到JSON文件
    with open(args.output, 'w', encoding='utf-8') as f:
        # 需要先处理Image对象，因为它不能被序列化
        serializable_results = results.copy()
        serializable_results.pop("query_analysis", None)
        
        # 确保我们可以序列化所有内容
        for i, result in enumerate(serializable_results.get("results", [])):
            if "image" in result:
                result.pop("image")
        
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"查询分析结果已保存到: {args.output}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        traceback.print_exc()