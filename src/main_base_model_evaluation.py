import os
from typing import List
from config_loader.models import EvaluationConfig, LLMConfig, LogConfig
from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import BaselineEvaluationModeExecution
from logger_manager import LoggerManager, LoggerMixin

from utils.evaluation_mode_validator import EvaluationModeValidator
from utils.llm_manager import LLMManager

DATA_FOLDER_PATH ="data"

class Main(LoggerMixin):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        # self.llm = LLMS[model_name]

        
    def run(self):
        llm_config = LLMConfig(
            model_name= self.model_name,
            # base_url = "http://localhost:11434",
            embedding_model_name= "snowflake-arctic-embed2",
            request_timeout= 600.0,
            temperature= 0.3
        )
        validator_llm_config = LLMConfig(
            model_name ="gemma3:12b", 
            # base_url="http://156.35.95.18:11434",
            embedding_model_name= "snowflake-arctic-embed2",
            request_timeout= 600.0,
        )
        evaluation_config = EvaluationConfig(
            questions_file_path = "dataset/questions/1-questions_persons.txt",
            prompts_file_path= "dataset/questions/prompts/1-questions_persons_prompts.txt",
            answers_file_path= "dataset/answers/1-questions_persons_answers.txt",
            results_folder_path= "results",
            reports_folder_path= "reports"
        )
        
        LoggerManager.initialize("INFO")
        LLMManager().init(llm_config)
        llm = LLMManager.create_llm_by_config(llm_config)
        EvaluationModeValidator().init(validator_llm_config)
        try:
            documents = StaticDataProcessor.load_pdf_documents(DATA_FOLDER_PATH)
            execution = BaselineEvaluationModeExecution(llm, evaluation_config, documents)

            execution.run()
        except Exception as e:  # Captura cualquier otra excepción
            self.logger.info("An unexpected error occurred:", e)
    



if __name__ == "__main__":
    models_name = [
        "llama3.2:3b", 
        # "gemma3:4b", 
        # "qwen3:4b",
        # "phi3:3.8b",
    ]

    Main(models_name[0]).run()