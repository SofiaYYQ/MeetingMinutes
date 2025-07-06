from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional, Union
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo

class ExecuteMode(str, Enum):
    CHAT = 'chat'
    EVALUATE = 'evaluate'
    NORMAL = 'normal'

class ResponseMode(str, Enum):
    BASIC = "basic"
    WITH_REASONING = "reasoning"

class DefaultSynthesizerConfig(BaseModel):
    type: Literal["default"]
    choice_description: Optional[str] = None

class SummarizerSynthesizerConfig(BaseModel):
    type: Literal["summarizer"]
    choice_description: Optional[str] = None
    summary_template_path: Optional[str] = None
    text_qa_template_path: Optional[str] = None
    bias_corrector_text_qa_template_path: Optional[str] = None

class RouterSynthesizerConfig(BaseModel):
    type: Literal["router"]
    synthesizers: List["SynthesizerConfig"]
    choice_description: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_synthesizers_choices(self):
        for s in self.synthesizers:
            if not s.choice_description:
                raise ValueError(f"The '{s.type}' synthesizer must have a non-empty 'choice_description' value within the 'router' synthesizer.")
        return self
    
SynthesizerConfig = Union[DefaultSynthesizerConfig, SummarizerSynthesizerConfig, RouterSynthesizerConfig]

# Resolver referencia recursiva
RouterSynthesizerConfig.model_rebuild()

class NLPKeywordsExtractorConfig(BaseModel):
    type: Literal["nlp"]
    model: Optional[str] = "es_core_news_md"

class KeyBertKeywordsExtractorConfig(BaseModel):
    type: Literal["keybert"]
    model: Optional[str] = "distiluse-base-multilingual-cased-v2"
    stop_words_path: Optional[str] = "config/stop_words.txt"

class KeywordsExtractorsConfig(BaseModel):
    type: Literal["multi_extractor"]
    extractors: List["KeywordsExtractorConfig"]

KeywordsExtractorConfig = Union[NLPKeywordsExtractorConfig, KeyBertKeywordsExtractorConfig, KeywordsExtractorsConfig]

# Resolver referencia recursiva
KeywordsExtractorsConfig.model_rebuild()

class LLMConfig(BaseModel):
    # use_server: bool
    base_url: Optional[str] = None
    model_name: str
    embedding_model_name: str
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


class DBConfig(BaseModel):
    database_path: str
    collection_name: str

class GeneralConfig(BaseModel):
    execute_mode: ExecuteMode
    llm: LLMConfig
    db: DBConfig
    data_folder_path: str

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

class ChunksConfig(BaseModel):
    chunk_size: int = Field(ge=0)
    chunk_overlap: int = Field(ge=0)

class PreRetrievalConfig(BaseModel):
    get_metadata: bool
    metadata_config: MetadataConfig
    use_chunks: bool
    chunks_config: ChunksConfig

class RetrievalConfig(BaseModel):
    # retriever: str
    use_keywords: bool
    keywords_extractor: KeywordsExtractorConfig
    response_synthesizer: SynthesizerConfig = Field(..., discriminator="type")

class PostRetrievalConfig(BaseModel):
    response_mode: ResponseMode

class AppConfig(BaseModel):
    log: LogConfig
    general: GeneralConfig
    evaluation_config: EvaluationConfig
    pre_retrieval: PreRetrievalConfig
    retrieval: RetrievalConfig
    post_retrieval: PostRetrievalConfig

class FullConfig(BaseModel):
    app: AppConfig