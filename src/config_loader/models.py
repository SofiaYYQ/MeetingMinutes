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
    output: Optional[str] = None

class CompositeStepModel(BaseStepModel):
    step_type: Literal["composite"]
    steps: List["StepModel"]

class IfStepModel(BaseStepModel):
    step_type: Literal["if"]
    condition: str
    if_true: "StepModel"
    if_false: Optional["StepModel"] = None

class LLMCallStepModel(BaseStepModel):
    step_type: Literal["llm_call"]
    prompt: str
    json_output: Optional[bool] = False

class FormatDocumentsActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["format_documents"]
    inputs: List[str]

class FormatDocumentActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["format_document"]
    inputs: List[str]

class ApplyFiltersActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["apply_filters"]
    inputs: List[str]

class CheckTermsInTextActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["check_terms_in_text"]
    inputs: List[List[str] | str]

class GoToStepModel(BaseStepModel):
    step_type: Literal["go_to"]
    target_id: str

class SetVariableStepModel(BaseStepModel):
    step_type: Literal["set_variable"]
    source: str

class ForEachStepModel(BaseStepModel):
    step_type: Literal["for_each"]
    iterate_obj: str
    step: "StepModel"
    collected_field: Optional[str] = None

class FormatListActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["format_list"]
    inputs: List[str]
    format_template: str
    separator: Optional[str] = "\n"

class AddToMemoryActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["add_to_memory"]
    name: str
    description: str
    result: str

class FormatMemoryActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["format_memory"]

class EvaluateActionStepModel(BaseStepModel):
    step_type: Literal["action"]
    action: Literal["evaluate"]
    max_intents: int = Field(default=5, gt=0)
    step: LLMCallStepModel
    condition:str
    prompt: str
    json_output: Optional[bool] = False
    
StepModel = Union[
    LLMCallStepModel, 
    FormatDocumentsActionStepModel, 
    FormatDocumentActionStepModel,
    CompositeStepModel, 
    IfStepModel, 
    ApplyFiltersActionStepModel, 
    CheckTermsInTextActionStepModel,
    GoToStepModel,
    SetVariableStepModel,
    ForEachStepModel,
    FormatListActionStepModel,
    AddToMemoryActionStepModel,
    FormatMemoryActionStepModel,
    EvaluateActionStepModel
]

CompositeStepModel.model_rebuild()
IfStepModel.model_rebuild()
ForEachStepModel.model_rebuild()
# class StepWrapper(BaseModel):
#     step: StepModel

class AppConfig(BaseModel):
    log: LogConfig
    general: GeneralConfig
    evaluation_config: EvaluationConfig
    data_processing: DataProcessingConfig
    workflow: List[StepModel]

class FullConfig(BaseModel):
    app: AppConfig



