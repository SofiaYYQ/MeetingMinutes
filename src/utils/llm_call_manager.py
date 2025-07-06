import json
import re
from llama_index.core.vector_stores.types import VectorStoreInfo
from metaclasses import SingletonMeta
from utils.llm_manager import LLMManager
from llama_index.core.llms import LLM

from utils.utils import Utils

MODELS_WITH_THINKING = ["qwen3:4b"]

class LLMCallManager(metaclass=SingletonMeta):
    
    def init(self):
       self.llm = LLMManager().create_json_output_llm()

    def get_document_all_metadata(self, vector_store_info:VectorStoreInfo, document:str):
        metadata_infos = vector_store_info.metadata_info
        strs_to_prompt = []
        for i, metadata_info in enumerate(metadata_infos):
            str_to_prompt = f"{i+1}) '{metadata_info.name}' ({metadata_info.description})"
            strs_to_prompt.append(str_to_prompt)

        prompt = f"Extrae información del contexto en formato JSON con {len(metadata_infos)} claves: {', '.join(strs_to_prompt)}. No incluyas ningún texto adicional fuera del JSON. Contexto: {document}"
        response = self.llm.complete(prompt)
        result = json.loads(response.text)
        return result
    
#     @staticmethod
#     def get_document_all_metadata_by_custom_llm(llm: LLM, vector_store_info:VectorStoreInfo, document:str):
#         metadata_infos = vector_store_info.metadata_info
#         metadata_fields = [m.name for m in metadata_infos]
#         strs_to_prompt = []
#         for i, metadata_info in enumerate(metadata_infos):
#             str_to_prompt = f"{i+1}) '{metadata_info.name}' (De tipo {metadata_info.type}. {metadata_info.description})"
#             strs_to_prompt.append(str_to_prompt)

#         keys_str='\n'.join(strs_to_prompt)
#         has_all_metadata = False
#         while not has_all_metadata:
            
#             prompt = f"""
#             Extrae información del contexto en formato JSON con {len(metadata_infos)} claves: 
#             {keys_str}
            
#             No incluyas ningún texto adicional fuera del JSON. 
#             Contexto: 
#             {document}

# """         
#             response = llm.complete(prompt)
#             result = json.loads(response.text)
#             has_all_metadata = Utils.has_required_fields(result, metadata_fields)
        
#         return result

    @staticmethod
    def get_document_all_metadata_by_custom_llm(llm: LLM, vector_store_info: VectorStoreInfo, document: str):
        metadata_infos = vector_store_info.metadata_info
        result = {}

        for i, metadata_info in enumerate(metadata_infos):
            field_name = metadata_info.name
            field_type = metadata_info.type
            field_description = metadata_info.description

            prompt = f"""
    Extrae el valor del siguiente campo de metadatos a partir del contexto proporcionado.

    Campo: '{field_name}'  
    Tipo: {field_type}  
    Descripción: {field_description}

    Devuelve únicamente un JSON con esta estructura:
    {{ "{field_name}": valor }}

    No incluyas ningún texto adicional fuera del JSON.

    Contexto:
    {document}
    """
            if field_type == "list":
                valid_list = False
                while not valid_list:
                    response = llm.complete(prompt)
                    partial_result = json.loads(response.text)
                    if type(partial_result[field_name]) == list:
                        valid_list = True
                        if field_name in partial_result:
                            result[field_name] = partial_result[field_name]
            else:        
                # Llamada al LLM
                response = llm.complete(prompt)
                partial_result = json.loads(response.text)
                if field_name in partial_result:
                    result[field_name] = partial_result[field_name]

        return result

    @staticmethod
    def process_complete_response(model_name, complete_response):
        if model_name in MODELS_WITH_THINKING:
            result_str = str(complete_response)
            return re.sub(r'<think>.*?</think>', '', result_str, flags=re.DOTALL).strip('\n')
        else:
            return str(complete_response)