import asyncio
import os
from typing import List
# from executions.executions import ChatModeExecution, EvaluationModeExecution, QueryModeExecution
# from config_loader.config_loader import ConfigLoader
# from data_processors.data_processor import DataProcessor
# from query_engines.creators import BasicQueryEngineCreator, ChatEngineCreator, CustomQueryEngineCreator, RetrieverQueryEngineCreator, RouterRetrieverQueryEngineCreator
# from response_processors.query_response_processor import BasicQueryResponseProcessor, JSONTextQueryResponseProcessor
# from rag_manager import RAGManager

from config_loader.models import EvaluationConfig, LLMConfig
# from config_loader.models import EvaluationConfig, ExecuteMode, KeyBertKeywordsExtractorConfig, KeywordsExtractorsConfig, LLMConfig, NLPKeywordsExtractorConfig, ResponseMode
# from config_loader.builders import KeywordsExtractorCreator, SynthesizerCreator
from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import WorkflowEvaluationModeExecution
from logger_manager import LoggerManager, LoggerMixin
# from query_engines.key_extractors import KeywordsExtractors, KeywordsKeyBertExtractor, KeywordsNLPExtractor
# from query_engines.key_extractors import KeywordsExtractorCreator
# from query_engines.key_extractors import KeywordsNLPExtractor
from qwen_workflow import QwenDocumentsBasedQAFlow
from utils.evaluation_mode_validator import EvaluationModeValidator
# from utils.llm_call_manager import LLMCallManager
# from llama_index.core import VectorStoreIndex, Settings, StorageContext
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama

from utils.llm_call_manager import LLMCallManager
from utils.llm_manager import LLMManager

from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader

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
        # MetadataInfo(
        #     name="hora_inicio",
        #     type="str",
        #     description=(
        #         "Es la hora en la que se inició la reunión. Está en formato HH:MM"
        #     ),
        # ),
        # MetadataInfo(
        #     name="hora_fin",
        #     type="str",
        #     description=(
        #         "Es la hora en la que se terminó la reunión. Está en formato HH:MM"
        #     ),
        # ),
        # MetadataInfo(
        #     name="lugar",
        #     type="str",
        #     description=(
        #         "Es el lugar donde se celebró la reunión."
        #     ),
        # ),
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
                "Es el nombre del secretorio o secretaria de la reunión."
            ),
        ),
    ],
)

DATA_FOLDER_PATH = "data"
# def load_documents()->List[Document]:
#     pdf_loader = PDFReader(return_full_document=True)
#     documents = []
#     for filename in os.listdir(DATA_FOLDER_PATH):
#         if filename.endswith(".pdf"):
#             filepath = os.path.join(DATA_FOLDER_PATH, filename)
#             docs = pdf_loader.load_data(file=filepath)
#             documents.extend(docs)
#             # docs_text = " ".join([doc.text for doc in docs])
#             # documents.append(Document(text=docs_text))
#     return documents

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
            base_url="http://156.35.95.18:11434", 
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
        # llm_json_output = LLMManager.create_json_output_llm_by_config(LLMConfig(
        #     model_name= "llama3.2",
        #     embedding_model_name= "snowflake-arctic-embed2",
        # ))
        metadata_llm_json_output = Ollama(
            model="llama3.2", 
            base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0, 
            json_mode=True
        )
        
        workflow_llm = Ollama(
            model="qwen3:4b", 
            base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0
        )
        workflow_llm_json_output = Ollama(
            model="qwen3:4b", 
            base_url="http://156.35.95.18:11434", 
            temperature=0.3, 
            request_timeout=600.0, 
            json_mode=True
        )
    
        # key_extractor = KeywordsExtractorCreator.create(
        #     KeywordsExtractorsConfig(
        #         type="multi_extractor", 
        #         extractors=[NLPKeywordsExtractorConfig(type="nlp"), KeyBertKeywordsExtractorConfig(type="keybert")]
        #     )
        # )

        try:
            documents = StaticDataProcessor.load_pdf_documents(DATA_FOLDER_PATH)
            for d in documents:
                metadata = LLMCallManager.get_document_all_metadata_by_custom_llm(metadata_llm_json_output, VECTOR_STORE_INFO, d.text)
                d.metadata.update(metadata)
                doc_filename = d.metadata["file_name"]
                self.logger.info(f"Extracted Metadata for the document {doc_filename}: {metadata}")
            # flow = DocumentsBasedQAFlow(timeout=60, verbose=True)
            flow = QwenDocumentsBasedQAFlow(workflow_llm, workflow_llm_json_output, timeout=600, verbose=True)
            execution = WorkflowEvaluationModeExecution(flow, evaluation_config, documents)
           
            asyncio.run(execution.run())
        except Exception as e:  # Captura cualquier otra excepción
            self.logger.info("An unexpected error occurred:", e)

    
    
if __name__ == "__main__":
    Main().run()