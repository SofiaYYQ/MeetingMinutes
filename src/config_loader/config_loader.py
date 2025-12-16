

from logger_manager import LoggerMixin
from utils.file_handler import FileHandler
from config_loader.models import FullConfig
from config_loader.models import EvaluationConfig, LLMConfig
from pydantic import ValidationError
from llama_index.core.vector_stores.types import VectorStoreInfo


class ConfigLoader(LoggerMixin):
    def __init__(self, path:str="config/settings.yml"):
        super().__init__()
        self.path = path
        self.config = None
        self._load()

    def _load(self):
        raw = FileHandler.read_yaml(self.path)
        try:
            self.config = FullConfig(**raw)
        except ValidationError as e:
            self.logger.info(f"Error al validar YAML: {e}")
            raise
    
    def get_config(self):
        return self.config
    
    def get_data_folder_path(self) -> str: 
        return self.config.app.data_processing.data_folder_path
    
    def get_metadata_config(self) -> VectorStoreInfo: 
        return self.config.app.data_processing.metadata_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        return self.config.app.evaluation_config
    
    def get_llm_config(self) -> LLMConfig:
        return self.config.app.general.llm
    
    def get_log_level(self) -> str:
        return self.config.app.log.log_level