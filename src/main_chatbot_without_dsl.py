
import asyncio
import re 
from data_processors.static_data_processor import StaticDataProcessor
from executions.workflow_executions import WorkflowEvaluationModeExecution
from main_workflow_without_dsl import DATA_FOLDER_PATH, VECTOR_STORE_INFO
from qwen_workflow import QwenDocumentsBasedQAFlow
import streamlit as st

from config_loader.config_loader import ConfigLoader
from config_loader.models import EvaluationConfig, ExecuteMode, LLMConfig
from logger_manager import LoggerManager
from utils.evaluation_mode_validator import EvaluationModeValidator
from utils.llm_call_manager import LLMCallManager
from utils.llm_manager import LLMManager
from llama_index.llms.ollama import Ollama

@st.cache_resource
def get_rag_manager():

    # # settings_path = "config/settings-evaluate-gemma3.12b-basic.yml"
    # # settings_path = "config/settings-evaluate-gemma3.12b-reasoning.yml"
    # # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-basic.yml"
    # # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-reasoning.yml"
    # # settings_path = "config/settings-evaluate-gemma3.12b-metadata-basic.yml"
    # settings_path = "config/settings-chat-gemma3.12b-metadata-basic.yml"
    
    # # settings_path = "config/settings-evaluate-llama3.2-basic.yml"
    
    # config_loader = ConfigLoader(settings_path)
    # full_config = config_loader.get_config()

    # # Log settings
    # LoggerManager.initialize(full_config.app.log.log_level)

    # processor = None
    # engine_creator = None

    # response_mode = full_config.app.post_retrieval.response_mode
    # execute_mode = full_config.app.general.execute_mode
    # get_metadata = full_config.app.pre_retrieval.get_metadata
    # metadata_config = full_config.app.pre_retrieval.metadata_config


    # if response_mode == ResponseMode.BASIC:
    #     processor = BasicQueryResponseProcessor()
    # elif response_mode == ResponseMode.WITH_REASONING:
    #     processor = JSONTextQueryResponseProcessor()

    # if execute_mode == ExecuteMode.CHAT:
    #     # engine_creator = ChatEngineCreator()
    #     engine_creator = RouterRetrieverQueryEngineCreator(metadata_config, KeywordsExtractors([KeywordsNLPExtractor(), KeywordsKeyBertExtractor()]))
    # elif execute_mode == ExecuteMode.EVALUATE or execute_mode == ExecuteMode.NORMAL:
    #     if not get_metadata:
    #         engine_creator = BasicQueryEngineCreator()
    #         # data_processor = BasicDataProcessor()
    #     else:
    #         #TODO: meter al fichero de configuracion los extractores
    #         engine_creator = RouterRetrieverQueryEngineCreator(metadata_config, KeywordsExtractors([KeywordsNLPExtractor(), KeywordsKeyBertExtractor()]))
    #         # data_processor = GetMetadataDataProcessor(metadata_config)
    
    # data_processor = DataProcessor(
    #         full_config.app.general.data_folder_path,
    #         full_config.app.pre_retrieval.get_metadata,
    #         full_config.app.pre_retrieval.metadata_config,
    #         full_config.app.pre_retrieval.use_chunks,
    #         full_config.app.pre_retrieval.chunks_config
    #     )

    # return RAGManager(processor, engine_creator, data_processor, full_config)

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

    
        
        # flow = DocumentsBasedQAFlow(timeout=60, verbose=True)
        flow = QwenDocumentsBasedQAFlow(workflow_llm, workflow_llm_json_output, timeout=600, verbose=True)

        return flow



# Inicialización
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = get_rag_manager()
    
header = "Chatbot para las Actas de Comunidad"
welcome_message = "Hola, escribe preguntas sobre las actas"
input_placeholder = "Pregunta sobre alguna persona"
thinking_message = "Pensando sobre tu pregunta..."
loading_message = "Cargando la información"

st.header(header)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        { 
            "role": "assistant",
            "content": welcome_message,
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text=loading_message):
        if "documents" not in st.session_state:
            metadata_llm_json_output = Ollama(
                model="llama3.2", 
                base_url="http://156.35.95.18:11434", 
                temperature=0.3, 
                request_timeout=600.0, 
                json_mode=True
            )
            documents = StaticDataProcessor.load_pdf_documents(DATA_FOLDER_PATH)
            for d in documents:
                metadata = LLMCallManager.get_document_all_metadata_by_custom_llm(metadata_llm_json_output, VECTOR_STORE_INFO, d.text)
                d.metadata.update(metadata)
                doc_filename = d.metadata["file_name"]
                print(f"Extracted Metadata for the document {doc_filename}: {metadata}")
            st.session_state.documents = documents

load_data()

def clean_message(self, text):
    # Eliminar contenido entre <think> y </think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Reemplazar URLs por localhost (por ejemplo, cambiar https://ejemplo.com/documento.pdf a http://localhost/documento.pdf)
    text = re.sub(r"https?://[^\s)]+", lambda match: match.group(0).replace(match.group(0).split('/')[2], "localhost"), text)

# load_data()
# chat_engine = st.session_state.rag_manager.query_engine

if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


async def run_query(prompt):
    response = await st.session_state.rag_manager.run(
        documents=st.session_state.documents,
        query=prompt
    )
    return response

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(thinking_message):
            print(f"Pregunta: {prompt}")
            
            # transformed_query = st.session_state.rag_manager.query_response_processor.transform(prompt)
            # # question_with_prompt = f"{prompts[index]} Pregunta: {question}" 
            # response = st.session_state.rag_manager.query_engine.query(transformed_query)  # Ejecutar consulta
            # response_text = st.session_state.rag_manager.query_response_processor.process(response)
            # response = await st.session_state.rag_manager.run(
            #     documents=st.session_state.documents,
            #     query=prompt
            # )
            
            response = asyncio.run(run_query(prompt))
            # st.session_state.messages.append({"role": "assistant", "content": response})
            # st.markdown(response)

            # st.session_state.rag_manager.query_engine.query(transformed_query)
            
            st.write(response)
            print(f"Respuesta: {response}\n")

            # cleaned_content = clean_message(response.response)  # Aplicamos el filtro
            # st.write(cleaned_content)
            # print(cleaned_content)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)  # Add response to message history