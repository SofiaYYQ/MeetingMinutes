

from logger_manager import LoggerMixin
from utils.file_handler import FileHandler
from config_loader.models import FullConfig
from pydantic import ValidationError

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