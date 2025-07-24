import json
from types import SimpleNamespace
from typing import Any, List
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document

from config_loader.models import BaseStepModel, MetadataConfig
from config_loader.steps import StepFactory
from logger_manager import LoggerManager


from utils.llm_call_manager import LLMCallManager
from utils.utils import Utils

# pasar a utils
def get_metadata_info(metadata_config:MetadataConfig, metadata_name:str|None):
    for metadata in metadata_config.metadata_info:
        if metadata.name == metadata_name:
            return metadata
    return None

class DocumentsBasedQAFlowExecutor():
    def __init__(self, llm:Ollama, llm_json_output:Ollama, metadata_config:MetadataConfig, documents: List[Document], workflow: List[BaseStepModel]) -> None:
        self.logger = LoggerManager.get_logger(name=self.__class__.__name__)
        self.llm = llm
        self.llm_json_output = llm_json_output
        self.metadata_config = metadata_config
        self.documents = documents
        self.step_results = []
        self.workflow = workflow
        self.context = {}
        self.context["documents"] = documents
    
    def add_to_memory(self, name:str, description:str, result:str):
        self.step_results.append(
            {
                "name": name,
                "description": description,
                "result": result,
            }
        )
    def add_to_context(self, key:str, value:Any):
        self.context[key] = value

        
    def format_memory(self):
        return "".join(
            f"Paso {i}: {step['name']}\nDescripción: {step['description']}\nResultado: {step['result']}\n\n"
            for i, step in enumerate(self.step_results, 1)
        )


    def run(self, query:str)->str:
        self.context["query"] = query

        
        steps_by_id = {s.id: s for s in self.workflow}
        step_ids = list(steps_by_id.keys())
        current_step_index = 0
        output = None

        while current_step_index < len(step_ids):
            step_id = step_ids[current_step_index]
            s = steps_by_id[step_id]

            step = StepFactory.create(
                s,
                global_context=self.context,
                llm_call=self.get_llm_output,
                json_llm_call=self.get_valid_json_output,
                metadata_config=self.metadata_config,
                add_to_memory = self.add_to_memory,
                format_memory = self.format_memory,
                add_to_context = self.add_to_context
            )

            output = step.run()
            
            if isinstance(output, dict) and "go_to" in output:
                target_id = output["go_to"]
                if target_id in steps_by_id:
                    current_step_index = step_ids.index(target_id)
                    continue
                else:
                    raise ValueError(f"Destination step '{target_id}' not found.")
                
            # self.context[s.output] = output
            current_step_index += 1

        if output is None:
            context_values = list(self.context.values())
            output = context_values[-1] if context_values else "No output."

        self.reset()
        return output
    
    def reset(self):
        self.step_results = []
        self.context = {}
        self.context["documents"] = self.documents

    def get_llm_output(self, prompt):
        response = self.llm.complete(prompt)
        return response.text
    
    def process_complete_response(self, complete):
        return LLMCallManager.process_complete_response(self.llm.model, complete)
    
    def get_valid_json_output(self, prompt, keys=None):
        if keys == None:
            keys = Utils.extract_json_keys_from_text(prompt)
        while True:
            try:
                response = self.llm_json_output.complete(prompt)
                response_str = self.process_complete_response(response)
                data = json.loads(response_str)
                
                obj = SimpleNamespace(**data)

                if keys:
                    if all([key in data for key in keys]):
                        self.logger.info("Valid JSON: %s", data)
                        return obj
                else:
                    self.logger.info("Valid JSON: %s", data)
                    return obj
            except json.JSONDecodeError:
                self.logger.info("Invalid JSON. Try it again.")