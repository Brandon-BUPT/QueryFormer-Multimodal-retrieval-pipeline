"""
多模态检索系统管道模块
"""

from .base_pipeline import BasePipeline
from .registry import PipelineRegistry
from .factory import ComponentFactory
from .retrieval_pipeline import RetrievalPipeline

# 注册所有可用的管道
PipelineRegistry.register("retrieval")(RetrievalPipeline)

__all__ = [
    "BasePipeline",
    "PipelineRegistry",
    "ComponentFactory",
    "RetrievalPipeline",
]
