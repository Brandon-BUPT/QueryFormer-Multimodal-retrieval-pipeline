from typing import Dict, Type, Any
from .base_pipeline import BasePipeline

class PipelineRegistry:
    """Registry for different pipeline implementations."""
    
    _registry: Dict[str, Type[BasePipeline]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a pipeline class."""
        def inner_wrapper(wrapped_class: Type[BasePipeline]):
            if name in cls._registry:
                print(f"WARNING: Pipeline {name} already exists. Overwriting.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def get_pipeline(cls, name: str, config: Dict[str, Any]) -> BasePipeline:
        """Get a pipeline instance by name."""
        if name not in cls._registry:
            raise ValueError(f"Pipeline {name} not found in registry.")
        return cls._registry[name](config)
    
    @classmethod
    def list_pipelines(cls) -> Dict[str, Type[BasePipeline]]:
        """List all registered pipelines."""
        return cls._registry.copy() 