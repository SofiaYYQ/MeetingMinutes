import asyncio

from config_loader.models import EvaluationConfig, LLMConfig
from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import WorkflowEvaluationModeExecution
from logger_manager import LoggerManager, LoggerMixin

from qwen_workflow import QwenDocumentsBasedQAFlow
from utils.evaluation_mode_validator import EvaluationModeValidator

from utils.llm_call_manager import LLMCallManager
from utils.llm_manager import LLMManager

from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.llms.ollama import Ollama

VECTOR_STORE_INFO = VectorStoreInfo(
    content_info="actas de reuniones de la comunidad de vecinos",
    metadata_info=[
        MetadataInfo(
            name="fecha",
            type="str",
            description=(
                "Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA."
            ),
        ),
        MetadataInfo(
            name="num_asistentes",
            type="int",
            description=(
                "Es el número de asistentes a la reunión"
            ),
        ),
        MetadataInfo(
            name="lista_asistentes",
            type="list",
            description=(
                "Es la lista de nombres de los asistentes de la reunión."
            ),
        ),
        MetadataInfo(
            name="presidente",
            type="str",
            description=(
                "Es el nombre del presidente o presidenta de la reunión."
            ),
        ),
        MetadataInfo(
            name="secretario",
            type="str",
            description=(
                "Es el nombre del secretario o secretaria de la reunión."
            ),
        ),
    ],
)

DATA_FOLDER_PATH = "data"

class Main(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def run(self):
        llm_config = LLMConfig(
            model_name= "qwen3:4b",
            # base_url = "http://localhost:11434",
            embedding_model_name= "snowflake-arctic-embed2",
            request_timeout= 600.0,
            temperature= 0.3
        )
        validator_llm_config = LLMConfig(
            model_name ="llama3.2", 
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

        # Log settings
        LoggerManager.initialize("INFO")
        LLMManager().init(llm_config)
        EvaluationModeValidator().init(validator_llm_config)

        metadata_llm_json_output = Ollama(
            model="llama3.2", 
            # base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0, 
            json_mode=True
        )
        
        workflow_llm = Ollama(
            model="qwen3:4b", 
            # base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0
        )
        workflow_llm_json_output = Ollama(
            model="qwen3:4b", 
            # base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0, 
            json_mode=True
        )

        try:
            documents = StaticDataProcessor.load_pdf_documents(DATA_FOLDER_PATH)
            for d in documents:
                metadata = LLMCallManager.get_document_all_metadata_by_custom_llm(metadata_llm_json_output, VECTOR_STORE_INFO, d.text)
                d.metadata.update(metadata)
                doc_filename = d.metadata["file_name"]
                self.logger.info(f"Extracted Metadata for the document {doc_filename}: {metadata}")

            flow = QwenDocumentsBasedQAFlow(workflow_llm, workflow_llm_json_output, timeout=600, verbose=True)
            execution = WorkflowEvaluationModeExecution(flow, evaluation_config, documents)
           
            asyncio.run(execution.run())
        except Exception as e:
            self.logger.info("An unexpected error occurred:", e)

    
    
if __name__ == "__main__":
    Main().run()