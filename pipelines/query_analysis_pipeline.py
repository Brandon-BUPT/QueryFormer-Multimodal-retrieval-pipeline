from typing import Dict, Any, List
import torch
import numpy as np
from PIL import Image
import os

from .base_pipeline import BasePipeline
from .factory import ComponentFactory
from .retrieval_pipeline import RetrievalPipeline
from src.encoding.image_encoder import encode_image
from src.encoding.text_encoder import encode_text
from src.encoding.joint_encoder import encode_image_text
from src.query_analysis.query_analyzer import QueryAnalyzer


class QueryAnalysisPipeline(BasePipeline):
    """结合查询分析和检索的管道"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.retrieval_pipeline = self._initialize_retrieval_pipeline()
        print("🚀 QueryAnalysisPipeline initialized successfully!")

    def _initialize_components(self) -> Dict[str, Any]:
        """初始化查询分析组件"""
        components = {}
        
        print("⚙️ Initializing query analysis components...")
        
        # 初始化查询分析器
        components["query_analyzer"] = QueryAnalyzer(self.config.get("query_analyzer", {}))
        
        return components
    
    def _initialize_retrieval_pipeline(self) -> RetrievalPipeline:
        """初始化检索管道"""
        print("⚙️ Initializing retrieval pipeline...")
        from .registry import PipelineRegistry
        
        # 使用相同的配置初始化检索管道
        retrieval_config = self.config.copy()
        # 更新检索特定配置（如果有的话）
        retrieval_config.update(self.config.get("retrieval", {}))
        
        return PipelineRegistry.get_pipeline("retrieval", retrieval_config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行查询分析管道"""
        query_type = input_data.get("query_type", "multimodal2text")
        print(f"🔍 Running query analysis pipeline with mode: {query_type}")
        
        # 只对图文共查模式应用查询分析
        if query_type == "multimodal2text" and "image" in input_data and "text" in input_data:
            return self._analyze_and_retrieve(
                image=input_data["image"], 
                query_text=input_data["text"]
            )
        else:
            # 对于其他查询类型，直接使用检索管道
            print(f"不支持的查询类型或数据不完整，跳过查询分析步骤: {query_type}")
            return {
                "results": self.retrieval_pipeline.run(input_data),
                "query_analysis": None
            }
    
    def _analyze_and_retrieve(self, image: Image.Image, query_text: str) -> Dict[str, Any]:
        """分析查询并执行检索"""
        # 步骤1: 使用大模型分析查询
        print(f"Analyzing query: '{query_text}'")
        enhanced_query, analysis_result = self.components["query_analyzer"].get_enhanced_query(
            image=image, 
            query_text=query_text
        )
        
        # 步骤2: 使用增强的查询文本执行检索
        if analysis_result["success"]:
            print(f"使用增强查询执行检索: '{enhanced_query}'")
            # 如果分析成功，使用增强查询
            retrieval_results = self.retrieval_pipeline.run({
                "query_type": "multimodal2text",
                "image": image,
                "text": enhanced_query  # 使用增强查询
            })
        else:
            print(f"查询分析失败，使用原始查询: '{query_text}'")
            # 如果分析失败，使用原始查询
            retrieval_results = self.retrieval_pipeline.run({
                "query_type": "multimodal2text",
                "image": image,
                "text": query_text  # 使用原始查询
            })
        
        # 返回检索结果和查询分析信息
        return {
            "results": retrieval_results,
            "query_analysis": analysis_result,
            "original_query": query_text,
            "enhanced_query": enhanced_query if analysis_result["success"] else None,
            "keywords": {
                "explicit": analysis_result.get("analysis", {}).get("explicit_keywords", []) if analysis_result["success"] else [],
                "implicit": analysis_result.get("analysis", {}).get("implicit_keywords", []) if analysis_result["success"] else []
            }
        }