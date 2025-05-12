from src.data_preprocessing.preprocessor import Preprocessor
from src.encoding.joint_encoder import JointEncoder
from src.indexing.faiss_lsh import FaissLSH
from src.retrieval.retriever import Retriever

class ComponentFactory:
    @staticmethod
    def create_component(component_type: str, config: dict):
        components = {
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
        component_class = components[component_type][config["type"]]
        return component_class(config["params"])