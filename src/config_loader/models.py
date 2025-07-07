from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional, Union
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo

class ExecuteMode(str, Enum):
    CHAT = 'chat'
    EVALUATE = 'evaluate'
    NORMAL = 'normal'


class LLMConfig(BaseModel):
    # use_server: bool
    base_url: Optional[str] = None
    model_name: str
    # embedding_model_name: str
    request_timeout: Optional[float] = Field(default=None, ge=0.0)
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    def model_dump_for_create_llm(self):
        data = self.model_dump(exclude_none=True, exclude={"embedding_model_name"})
        data['model'] = data.pop('model_name')
        
        return data
    
    def model_dump_for_create_embed_model(self):
        data = self.model_dump(exclude_none=True, include={"base_url", "embedding_model_name"})
        data['model_name'] = data.pop('embedding_model_name')
        return data


class GeneralConfig(BaseModel):
    execute_mode: ExecuteMode
    llm: LLMConfig

class EvaluationConfig(BaseModel):
    questions_file_path: str
    prompts_file_path: str
    answers_file_path: str
    results_folder_path: str
    reports_folder_path: str

class LogConfig(BaseModel):
    log_level : Literal['INFO', 'WARN', "ERROR", "DEBUG", "CRITICAL"]

class MetadataConfig(VectorStoreInfo):
    metadata_info: list[MetadataInfo] = Field(alias="fields_info")
    content_info: str = Field(alias="data_description")

    # class Config:
    #     allow_population_by_field_name = True

# class ChunksConfig(BaseModel):
#     chunk_size: int = Field(ge=0)
#     chunk_overlap: int = Field(ge=0)

class DataProcessingConfig(BaseModel):
    data_folder_path: str
    metadata_config: MetadataConfig
    # chunks_config: ChunksConfig


class BaseStepModel(BaseModel):
    id: str
    output: str

class CompositeStepModel(BaseStepModel):
    steps: List["StepModel"]

class IfStepModel(BaseStepModel):
    condition: str
    if_true: "StepModel"
    if_false: Optional["StepModel"] = None

class LLMCallStepModel(BaseStepModel):
    prompt: str
    json_output: Optional[bool] = False

class FormatDocumentsActionStepModel(BaseStepModel):
    action: Literal["format_documents"]
    inputs: List[str]

class ApplyFiltersActionStepModel(BaseStepModel):
    action: Literal["apply_filters"]
    inputs: List[str]

class CheckTermsInTextActionStepModel(BaseStepModel):
    action: Literal["check_terms_in_text"]
    inputs: List[List[str] | str]

StepModel = Union[
    LLMCallStepModel, 
    FormatDocumentsActionStepModel, 
    CompositeStepModel, 
    IfStepModel, 
    ApplyFiltersActionStepModel, 
    CheckTermsInTextActionStepModel
]

CompositeStepModel.model_rebuild()
IfStepModel.model_rebuild()

class StepWrapper(BaseModel):
    step: StepModel

class AppConfig(BaseModel):
    log: LogConfig
    general: GeneralConfig
    evaluation_config: EvaluationConfig
    data_processing: DataProcessingConfig
    workflow: List[StepWrapper]

class FullConfig(BaseModel):
    app: AppConfig



