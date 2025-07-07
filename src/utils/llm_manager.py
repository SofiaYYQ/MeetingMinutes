
from config_loader.models import LLMConfig
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import Settings

from metaclasses import SingletonMeta

class LLMManager(metaclass=SingletonMeta):
    def init(self, llm_config:LLMConfig):
       self.llm_config = llm_config
       self._set_general_llm()

    def _set_general_llm(self):
        # Settings.llm = self.create_llm()
        # Settings.embed_model = self.create_embed_model()
        pass

    def create_llm(self):
        input_params = self.llm_config.model_dump_for_create_llm()
        return Ollama(**input_params)
    
    # def create_embed_model(self):
    #     input_params = self.llm_config.model_dump_for_create_embed_model()

    #     return OllamaEmbedding(**input_params)
        
    def create_json_output_llm(self):
        input_params = self.llm_config.model_dump_for_create_llm()
        json_params = {
            "temperature": 0.0,
            "json_mode": True,
        }

        filtered_params = input_params | json_params

        return Ollama(**filtered_params)
    
    @staticmethod
    def create_llm_by_config(llm_config:LLMConfig):
        input_params = llm_config.model_dump_for_create_llm()
        return Ollama(**input_params)
    
    @staticmethod
    def create_json_output_llm_by_config(llm_config:LLMConfig):
        input_params = llm_config.model_dump_for_create_llm()
        json_params = {
            # "temperature": 0.0,
            "json_mode": True,
        }

        filtered_params = input_params | json_params
        return Ollama(**filtered_params)