from typing import Dict, Any, Type
from src.data_preprocessing.preprocessor import Preprocessor
from src.encoding.joint_encoder import JointEncoder
from src.indexing.faiss_lsh import FaissLSH
from src.retrieval.retriever import Retriever

class ComponentFactory:
    """Factory for creating pipeline components."""
    
    _component_registry: Dict[str, Dict[str, Type]] = {
        "preprocessor": {
            "standard": Preprocessor
        },
        "encoder": {
            "joint": JointEncoder
        },
        "indexer": {
            "faiss_lsh": FaissLSH
        },
        "retriever": {
            "standard": Retriever
        }
    }
    
    @classmethod
    def register_component(cls, component_type: str, name: str, component_class: Type):
        """Register a new component."""
        if component_type not in cls._component_registry:
            cls._component_registry[component_type] = {}
        
        print(f"Registering component: {component_type}.{name}")
        cls._component_registry[component_type][name] = component_class
    
    @classmethod
    def get_registry(cls):
        """Get the current component registry."""
        return cls._component_registry
    
    @classmethod
    def create_component(cls, component_type: str, config: Dict[str, Any]):
        """Create a component instance based on type and configuration."""
        if component_type not in cls._component_registry:
            raise ValueError(f"Component type '{component_type}' not found in registry. Available types: {list(cls._component_registry.keys())}")
        
        component_name = config["type"]
        if component_name not in cls._component_registry[component_type]:
            available = list(cls._component_registry[component_type].keys())
            raise ValueError(f"Component '{component_name}' not found in {component_type} registry. Available components: {available}")
        
        print(f"Creating component: {component_type}.{component_name}")
        component_class = cls._component_registry[component_type][component_name]
        try:
            return component_class(config["params"])
        except Exception as e:
            print(f"Error creating component {component_type}.{component_name}: {str(e)}")
            raise