
import json
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores.types import VectorStoreInfo

from metaclasses import SingletonMeta
from config_loader.models import LLMConfig
from logger_manager import LoggerMixin

from utils.llm_manager import LLMManager

class EvaluationModeValidator(LoggerMixin, metaclass=SingletonMeta):
    def init(self, llm_config: LLMConfig = None):
       self.local_llm = LLMManager.create_json_output_llm_by_config(llm_config) if llm_config else LLMManager().create_json_output_llm()

    def compare(self, expected:str, real:str):
        pregunta = f"¿Contiene la lista resultante '{real}' los mismos que la lista esperada '{expected}' SIN IMPORTAR el ORDEN? Primero compara un elemento por un elemento y después contesta."
        prompt = f"Genera una respuesta estrictamente en formato JSON con dos claves: 1) 'razonamiento' y 2) 'respuesta' (solo 'Sí' o 'No'). No incluyas ningún texto adicional fuera del JSON. Pregunta: {pregunta}"

        # Validar si es json valido y si existe los dos claves que queremos
        response = self.local_llm.complete(prompt)
        result = json.loads(response.text)
        return result
    
    def get_formatted_answers(self, real:list[str], format_prompt:list[str])->list[str]:
        formatted_answers = []
        for answer, format in zip(real, format_prompt):

            prompt = f"Extrae respuesta únicamente de la información dada a continuación: <<< {answer} >>>. Debe estar estrictamente en formato JSON con una única clave 'respuesta' ({format}). No incluyas ningún texto adicional fuera del JSON."

            response = self.local_llm.complete(prompt)
            try:
                result = json.loads(response.text)
                self.logger.info(f"Resultado formateo: {result}")
                formatted_answer = result["respuesta"]
                if type(formatted_answer) == int:
                    formatted_answer = str(formatted_answer)
                formatted_answers.append(formatted_answer)
            except Exception as e:
                self.logger.info(f"Error: {type(e).__name__} - {e}. Cannot retrieve formatted answer.")
                formatted_answers.append(answer)

        return formatted_answers
    

