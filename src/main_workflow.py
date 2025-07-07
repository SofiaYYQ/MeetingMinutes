import asyncio
from config_loader.config_loader import ConfigLoader
from config_loader.models import EvaluationConfig, LLMConfig

from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import ExecutorEvaluationModeExecution, WorkflowEvaluationModeExecution
from logger_manager import LoggerManager, LoggerMixin
from new_workflow import DocumentsBasedQAFlowExecutor
from qwen_workflow import QwenDocumentsBasedQAFlow
from utils.evaluation_mode_validator import EvaluationModeValidator


from utils.llm_call_manager import LLMCallManager
from utils.llm_manager import LLMManager

from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.llms.ollama import Ollama

class Main(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def run(self):
        config_loader = ConfigLoader()
        full_config = config_loader.get_config()
        self.full_config = full_config
    
        # llm_config = LLMConfig(
        #     model_name= "qwen3:4b",
        #     # base_url = "http://localhost:11434",
        #     embedding_model_name= "snowflake-arctic-embed2",
        #     request_timeout= 600.0,
        #     temperature= 0.3
        # )
        validator_llm_config = LLMConfig(
            model_name ="llama3.2", 
            # base_url="http://156.35.95.18:11434", 
            # embedding_model_name= "snowflake-arctic-embed2",
            request_timeout= 600.0,
        )
        # evaluation_config = EvaluationConfig(
        #     questions_file_path = "dataset/questions/1-questions_persons.txt",
        #     prompts_file_path= "dataset/questions/prompts/1-questions_persons_prompts.txt",
        #     answers_file_path= "dataset/answers/1-questions_persons_answers.txt",
        #     results_folder_path= "results",
        #     reports_folder_path= "reports"
        # )
        evaluation_config = self.full_config.app.evaluation_config

        LoggerManager.initialize(full_config.app.log.log_level)
        LLMManager().init(self.full_config.app.general.llm)
        EvaluationModeValidator().init(validator_llm_config)
        # llm_json_output = LLMManager.create_json_output_llm_by_config(LLMConfig(
        #     model_name= "llama3.2",
        #     embedding_model_name= "snowflake-arctic-embed2",
        # ))
        llm_config_for_metadata = self.full_config.app.general.llm.model_copy()
        llm_config_for_metadata.model_name = "llama3.2"
        metadata_llm_json_output = LLMManager.create_json_output_llm_by_config(llm_config_for_metadata)
        # metadata_llm_json_output = Ollama(
        #     model="llama3.2", 
        #     # base_url="http://156.35.95.18:11434", 
        #     temperature=0.3, 
        #     request_timeout=600.0, 
        #     json_mode=True
        # )
        # workflow_llm = Ollama(
        #     model="qwen3:4b", 
        #     # base_url="http://156.35.95.18:11434", 
        #     temperature=0.3, 
        #     request_timeout=600.0
        # )
        # workflow_llm_json_output = Ollama(
        #     model="qwen3:4b", 
        #     # base_url="http://156.35.95.18:11434", 
        #     temperature=0.3, 
        #     request_timeout=600.0, 
        #     json_mode=True
        # )
        workflow_llm = LLMManager.create_llm_by_config(self.full_config.app.general.llm)
        workflow_llm_json_output = LLMManager.create_json_output_llm_by_config(self.full_config.app.general.llm)
        # key_extractor = KeywordsExtractorCreator.create(
        #     KeywordsExtractorsConfig(
        #         type="multi_extractor", 
        #         extractors=[NLPKeywordsExtractorConfig(type="nlp"), KeyBertKeywordsExtractorConfig(type="keybert")]
        #     )
        # )
        data_folder_path = self.full_config.app.data_processing.data_folder_path
        metadata_config = self.full_config.app.data_processing.metadata_config
        workflow = self.full_config.app.workflow
        try:
            documents = StaticDataProcessor.load_pdf_documents(data_folder_path)
            # for d in documents:
            #     metadata = LLMCallManager.get_document_all_metadata_by_custom_llm(metadata_llm_json_output, metadata_config, d.text)
            #     d.metadata.update(metadata)
            #     doc_filename = d.metadata["file_name"]
            #     self.logger.info(f"Extracted Metadata for the document {doc_filename}: {metadata}")

            # flow = QwenDocumentsBasedQAFlow(workflow_llm, workflow_llm_json_output, timeout=600, verbose=True)
            # execution = WorkflowEvaluationModeExecution(flow, evaluation_config, documents)
           
            # asyncio.run(execution.run())
            executor = DocumentsBasedQAFlowExecutor(workflow_llm, workflow_llm_json_output, metadata_config, documents, workflow)
            execution = ExecutorEvaluationModeExecution(executor, evaluation_config, documents)
            execution.run()
        except Exception as e:  # Captura cualquier otra excepción
            self.logger.info("An unexpected error occurred:", e)

    
    
if __name__ == "__main__":
    Main().run()