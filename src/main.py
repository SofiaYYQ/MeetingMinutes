from executions.executions import ChatModeExecution, EvaluationModeExecution, QueryModeExecution
from config_loader.config_loader import ConfigLoader
from data_processors.data_processor import DataProcessor
from query_engines.creators import BasicQueryEngineCreator, ChatEngineCreator, CustomQueryEngineCreator, RetrieverQueryEngineCreator, RouterRetrieverQueryEngineCreator
from response_processors.query_response_processor import BasicQueryResponseProcessor, JSONTextQueryResponseProcessor
from rag_manager import RAGManager
from config_loader.models import ExecuteMode, ResponseMode
from config_loader.builders import KeywordsExtractorCreator, SynthesizerCreator
from logger_manager import LoggerManager, LoggerMixin
# from query_engines.key_extractors import KeywordsExtractors, KeywordsKeyBertExtractor, KeywordsNLPExtractor
# from query_engines.key_extractors import KeywordsExtractorCreator
from query_engines.key_extractors import KeywordsNLPExtractor
from utils.evaluation_mode_validator import EvaluationModeValidator
from utils.llm_call_manager import LLMCallManager
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from utils.llm_manager import LLMManager

class Main(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def run(self):
        # settings_path = "config/settings-evaluate-gemma3.12b-basic.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-reasoning.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-basic.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-reasoning.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-basic.yml"
        settings_path = "config/settings-evaluate-gemma3.12b-metadata-reasoning.yml"
        
        # settings_path = "config/settings-evaluate-llama3.2-basic.yml"
        
        config_loader = ConfigLoader(settings_path)
        full_config = config_loader.get_config()
        self.full_config = full_config
        
        # Log settings
        LoggerManager.initialize(full_config.app.log.log_level)

        LLMManager().init(full_config.app.general.llm)
        LLMCallManager().init()
        EvaluationModeValidator().init()



        processor = None
        engine_creator = None
        
        response_mode = full_config.app.post_retrieval.response_mode
        execute_mode = full_config.app.general.execute_mode
        get_metadata = full_config.app.pre_retrieval.get_metadata
        metadata_config = full_config.app.pre_retrieval.metadata_config
        use_keywords = full_config.app.retrieval.use_keywords
        keywords_extractor_config = full_config.app.retrieval.keywords_extractor

        if response_mode == ResponseMode.BASIC:
            processor = BasicQueryResponseProcessor()
        elif response_mode == ResponseMode.WITH_REASONING:
            processor = JSONTextQueryResponseProcessor()

        if use_keywords:
            keywords_extractor = KeywordsExtractorCreator.create(keywords_extractor_config)
        else:
            # TODO: deberia ser no usar.
            keywords_extractor = None
        
        if execute_mode == ExecuteMode.CHAT:
            engine_creator = ChatEngineCreator()
        elif execute_mode == ExecuteMode.EVALUATE or execute_mode == ExecuteMode.NORMAL:
            if not get_metadata:
                # TODO: modificar esto tambien
                engine_creator = BasicQueryEngineCreator()
                # data_processor = BasicDataProcessor()
            else:
                synthesizer_config = full_config.app.retrieval.response_synthesizer
                synthesizer = SynthesizerCreator.create(synthesizer_config)
                engine_creator = CustomQueryEngineCreator(metadata_config, synthesizer, keywords_extractor)
        

        data_processor = DataProcessor(
                full_config.app.general.data_folder_path,
                full_config.app.pre_retrieval.get_metadata,
                full_config.app.pre_retrieval.metadata_config,
                full_config.app.pre_retrieval.use_chunks,
                full_config.app.pre_retrieval.chunks_config
            )
        


        rag_manager = RAGManager(processor, engine_creator, data_processor, full_config)
                
        try:
            rag_manager.init()
            

            execution = None
            if execute_mode == ExecuteMode.CHAT:
                execution = ChatModeExecution(rag_manager)
            elif execute_mode == ExecuteMode.EVALUATE:
                execution = EvaluationModeExecution(rag_manager, full_config)
            elif execute_mode == ExecuteMode.NORMAL:
                execution = QueryModeExecution(rag_manager)

            execution.run()
        except Exception as e:  # Captura cualquier otra excepción
            self.logger.info("An unexpected error occurred:", e)

if __name__ == "__main__":
    Main().run()