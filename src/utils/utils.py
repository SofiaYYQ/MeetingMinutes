
import datetime
import json
import re
from typing import Dict, List
from config_loader.models import MetadataConfig


class Utils():
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

    @staticmethod
    def get_metadata_info(metadata_config:MetadataConfig, metadata_name:str|None):
        for metadata in metadata_config.metadata_info:
            if metadata.name == metadata_name:
                return metadata
        return None
    
    @staticmethod
    def extract_json_keys_from_text(text:str)-> None | list[str]:
        json_match = re.search(r'\{\{.*?\}\}', text, re.DOTALL)
        found_keys = None
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str[1:-1]
            try:
                json_obj = json.loads(json_str)
                found_keys = list(json_obj.keys())
            except json.JSONDecodeError:
                pass

        return found_keys
