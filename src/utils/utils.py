
import datetime
from typing import Dict, List
from config_loader.models import FullConfig


class Utils():
    @staticmethod
    def get_analysis_output_name(prefix: str, full_config: FullConfig) -> str:
        model_name = full_config.app.general.llm.model_name
        get_metadata = full_config.app.pre_retrieval.get_metadata
        use_chunks = full_config.app.pre_retrieval.use_chunks
        chunks_config = full_config.app.pre_retrieval.chunks_config
        response_mode = full_config.app.post_retrieval.response_mode
    
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        model_name_in_file = model_name.replace(":", ".")

        chunks_indicator = f"{chunks_config.chunk_size}_{chunks_config.chunk_overlap}"
        
        parts = []
        parts.append(prefix)
        parts.append(f"-{model_name_in_file}")
        if get_metadata:
            parts.append(f"-metadata")
        if use_chunks:
            parts.append(f"-{chunks_indicator}")
        parts.append(f"-{response_mode.value}")
        parts.append(f"-{current_datetime}")
        
        # filename = f"{prefix}-{model_name_in_file}-{metadata_indicator}{chunks_indicator}-{response_mode.value}-{current_datetime}"
        filename = "".join(parts)
        return filename
    
    @staticmethod
    def get_specific_chromadb_name(full_config: FullConfig) -> str:
        get_metadata = full_config.app.pre_retrieval.get_metadata
        use_chunks = full_config.app.pre_retrieval.use_chunks
        chunks_config = full_config.app.pre_retrieval.chunks_config
        
        chunks_indicator = f"{chunks_config.chunk_size}_{chunks_config.chunk_overlap}"
        
        parts = []
        if get_metadata:
            parts.append(f"-metadata")
        if use_chunks:
            parts.append(f"-{chunks_indicator}")

        suffix = "".join(parts)

        database_path = full_config.app.general.db.database_path
        if '/' in database_path:
            separator_index = database_path.find('/')
            first_part = database_path[:separator_index]
            rest_of_path = database_path[separator_index:]
            return f"{first_part}{suffix}{rest_of_path}"
        else:
            return database_path

    @staticmethod
    def get_testing_analysis_output_name(prefix: str) -> str:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = []
        parts.append(prefix)
        parts.append("-prueba")
        parts.append(f"-{current_datetime}")
        
        filename = "".join(parts)
        return filename

    @staticmethod
    def get_custom_analysis_output_name(prefix: str, custom_name:str) -> str:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = []
        parts.append(prefix)
        parts.append(f"-{custom_name}")
        parts.append(f"-{current_datetime}")
        
        filename = "".join(parts)
        return filename
    
    @staticmethod
    def has_required_fields(json_obj: Dict, required_fields: List[str]) -> bool:
        return all(field in json_obj for field in required_fields)

