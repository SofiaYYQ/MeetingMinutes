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

            metadatas = [
                {'fecha': '24/02/2025', 'num_asistentes': 20, 'lista_asistentes': ['Juan Pérez Gutiérrez', 'Marta González Ramírez', 'Luis Ramírez Ortega', 'Ana Sánchez Herrera', 'Roberto Martínez Vázquez', 'Carmen Herrera Jiménez', 'Pedro Jiménez Suárez', 'Laura Díaz Castro', 'Manuel Ortega Medina', 'Isabel Castro Torres', 'Jorge Moreno Navarro', 'Beatriz Suárez Aguilar', 'Alejandro Torres Rojas', 'Natalia Vázquez Gutiérrez', 'Eduardo Rojas Martínez', 'Silvia Medina Pérez', 'Ricardo Flores Sánchez', 'Patricia Navarro Díaz', 'Daniel Gutiérrez Moreno', 'Rosa Aguilar Fernández'], 'presidente': 'Juan Pérez Gutiérrez', 'secretario': 'Rosa Aguilar Fernández'},
                {'fecha': '25/02/2025', 'num_asistentes': 20, 'lista_asistentes': ['Antonio Martínez López', 'Carmen Herrera Jiménez', 'Luis Ramírez Ortega', 'Isabel Castro Torres', 'Jorge Moreno Navarro', 'Beatriz Suárez Aguilar', 'Eduardo Rojas Martínez', 'Silvia Medina Pérez', 'Ricardo Flores Sánchez', 'Patricia Navarro Díaz', 'Daniel Gutiérrez Moreno', 'Rosa Aguilar Fernández', 'Francisco Torres Delgado', 'Laura Díaz Castro', 'Marta González Ramírez', 'Juan Pérez Gutiérrez', 'Ana Sánchez Herrera', 'Pedro Jiménez Suárez', 'Roberto Martínez Vázquez', 'Natalia Vázquez Gutiérrez'], 'presidente': 'Antonio Martínez López', 'secretario': 'Natalia Vázquez Gutiérrez'},
                {'fecha': '25/08/2025', 'num_asistentes': 18, 'lista_asistentes': ['Beatriz Suárez Aguilar', 'Manuel Ortega Medina', 'Isabel Castro Torres', 'Jorge Moreno Navarro', 'Patricia Navarro Díaz', 'Eduardo Rojas Martínez', 'Silvia Medina Pérez', 'Ricardo Flores Sánchez', 'Daniel Gutiérrez Moreno', 'Rosa Aguilar Fernández', 'Laura Díaz Castro', 'Marta González Ramírez', 'Antonio Martínez López', 'Alejandro Torres Rojas', 'Natalia Vázquez Gutiérrez', 'Francisco Torres Delgado', 'Pedro Jiménez Suárez', 'Ana Sánchez Herrera'], 'presidente': 'Beatriz Suárez Aguilar', 'secretario': 'Natalia Vázquez Gutiérrez'},
                {'fecha': '25/02/2026', 'num_asistentes': 17, 'lista_asistentes': ['Jorge Moreno Navarro', 'Laura Díaz Castro', 'Manuel Ortega Medina', 'Rosa Aguilar Fernández', 'Ricardo Flores Sánchez', 'Beatriz Suárez Aguilar', 'Pedro Jiménez Suárez', 'Ana Sánchez Herrera', 'Patricia Navarro Díaz', 'Eduardo Rojas Martínez', 'Silvia Medina Pérez', 'Francisco Torres Delgado', 'Daniel Gutiérrez Moreno', 'Natalia Vázquez Gutiérrez', 'Antonio Martínez López', 'Isabel Castro Torres', 'Marta González Ramírez'], 'presidente': 'Jorge Moreno Navarro', 'secretario': 'Natalia Vázquez Gutiérrez'},
                {'fecha': '25/08/2026', 'num_asistentes': 19, 'lista_asistentes': ['Manuel Ortega Medina', 'Beatriz Suárez Aguilar', 'Daniel Gutiérrez Moreno', 'Rosa Aguilar Fernández', 'Ricardo Flores Sánchez', 'Pedro Jiménez Suárez', 'Ana Sánchez Herrera', 'Patricia Navarro Díaz', 'Eduardo Rojas Martínez', 'Silvia Medina Pérez', 'Francisco Torres Delgado', 'Natalia Vázquez Gutiérrez', 'Antonio Martínez López', 'Isabel Castro Torres', 'Marta González Ramírez', 'Alejandro Torres Rojas', 'Jorge Moreno Navarro', 'Laura Díaz Castro', 'Carmen Herrera Jiménez'], 'presidente': 'Manuel Ortega Medina', 'secretario': 'Natalia Vázquez Gutiérrez'}
            ]
            for i, d in enumerate(documents):
                metadata = metadatas[i]
                d.metadata.update(metadata)
                doc_filename = d.metadata["file_name"]
                self.logger.info(f"Extracted Metadata for the document {doc_filename}: {metadata}")

            executor = DocumentsBasedQAFlowExecutor(workflow_llm, workflow_llm_json_output, metadata_config, documents, workflow)
            execution = ExecutorEvaluationModeExecution(executor, evaluation_config, documents)
            execution.run()
        except Exception as e:  # Captura cualquier otra excepción
            self.logger.info("An unexpected error occurred:", e)

    
    
if __name__ == "__main__":
    Main().run()