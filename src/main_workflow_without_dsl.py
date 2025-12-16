import asyncio

from config_loader.config_loader import ConfigLoader
from config_loader.models import LLMConfig
from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import WorkflowEvaluationModeExecution
from logger_manager import LoggerManager, LoggerMixin

from qwen_workflow import QwenDocumentsBasedQAFlow
from utils.evaluation_mode_validator import EvaluationModeValidator

from utils.llm_call_manager import LLMCallManager
from utils.llm_manager import LLMManager

from llama_index.llms.ollama import Ollama

class Main(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.config_loader = ConfigLoader()
    
    def run(self):
        llm_config = self.config_loader.get_llm_config()

        validator_llm_config = LLMConfig(
            model_name ="llama3.2", 
            # base_url="http://156.35.95.18:11434", 
            request_timeout= 1200.0,
        )
        
        # Log settings
        LoggerManager.initialize(self.config_loader.get_log_level())
        LLMManager().init(llm_config)
        EvaluationModeValidator().init(validator_llm_config)

        ollama_metadata_llm_json_output_args = {
            "model": "llama3.2",
            "temperature": llm_config.temperature,
            "request_timeout": llm_config.request_timeout
        }

        ollama_workflow_llm_args = {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature,
            "request_timeout": llm_config.request_timeout
        }
        
        if llm_config.base_url is not None:
            ollama_workflow_llm_args["base_url"] = llm_config.base_url
            ollama_metadata_llm_json_output_args["base_url"] = llm_config.base_url
        
        metadata_llm_json_output = Ollama(
            json_mode=True,
            **ollama_metadata_llm_json_output_args
        )

        workflow_llm = Ollama(
            **ollama_workflow_llm_args
        )

        workflow_llm_json_output = Ollama(
            json_mode=True, 
            **ollama_workflow_llm_args
        )

        try:
            documents = StaticDataProcessor.load_pdf_documents(self.config_loader.get_data_folder_path())
            for d in documents:
                metadata = LLMCallManager.get_document_all_metadata_by_custom_llm(
                    metadata_llm_json_output, 
                    self.config_loader.get_metadata_config(), 
                    d.text
                )
                d.metadata.update(metadata)
                doc_filename = d.metadata["file_name"]
                self.logger.info(f"Extracted Metadata for the document {doc_filename}: {metadata}")

            flow = QwenDocumentsBasedQAFlow(workflow_llm, workflow_llm_json_output, self.config_loader.get_metadata_config(), timeout=600, verbose=True)
            execution = WorkflowEvaluationModeExecution(
                flow, 
                self.config_loader.get_evaluation_config(), 
                documents
            )
           
            asyncio.run(execution.run())
        except Exception as e:
            self.logger.info("An unexpected error occurred:", e)

    
    
if __name__ == "__main__":
    Main().run()