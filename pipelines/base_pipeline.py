from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BasePipeline(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = self._initialize_components()

    @abstractmethod
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize pipeline components (e.g., preprocessor, encoder, indexer, retriever)."""
        pass

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the pipeline with input data (e.g., query, mode)."""
        pass