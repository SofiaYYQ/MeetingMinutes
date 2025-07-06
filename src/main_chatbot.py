
import re 
import streamlit as st

from query_engines.creators import BasicQueryEngineCreator, ChatEngineCreator, RetrieverQueryEngineCreator, RouterRetrieverQueryEngineCreator
from response_processors.query_response_processor import BasicQueryResponseProcessor, JSONTextQueryResponseProcessor
from rag_manager import RAGManager
from config_loader.config_loader import ConfigLoader
from config_loader.models import ExecuteMode, ResponseMode
from data_processors.data_processor import DataProcessor
from logger_manager import LoggerManager
from query_engines.key_extractors import KeywordsExtractors, KeywordsKeyBertExtractor, KeywordsNLPExtractor

@st.cache_resource
def get_rag_manager():
        # settings_path = "config/settings-evaluate-gemma3.12b-basic.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-reasoning.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-basic.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-512_0-reasoning.yml"
        # settings_path = "config/settings-evaluate-gemma3.12b-metadata-basic.yml"
        settings_path = "config/settings-chat-gemma3.12b-metadata-basic.yml"
        
        # settings_path = "config/settings-evaluate-llama3.2-basic.yml"
        
        config_loader = ConfigLoader(settings_path)
        full_config = config_loader.get_config()

        # Log settings
        LoggerManager.initialize(full_config.app.log.log_level)

        processor = None
        engine_creator = None

        response_mode = full_config.app.post_retrieval.response_mode
        execute_mode = full_config.app.general.execute_mode
        get_metadata = full_config.app.pre_retrieval.get_metadata
        metadata_config = full_config.app.pre_retrieval.metadata_config


        if response_mode == ResponseMode.BASIC:
            processor = BasicQueryResponseProcessor()
        elif response_mode == ResponseMode.WITH_REASONING:
            processor = JSONTextQueryResponseProcessor()

        if execute_mode == ExecuteMode.CHAT:
            # engine_creator = ChatEngineCreator()
            engine_creator = RouterRetrieverQueryEngineCreator(metadata_config, KeywordsExtractors([KeywordsNLPExtractor(), KeywordsKeyBertExtractor()]))
        elif execute_mode == ExecuteMode.EVALUATE or execute_mode == ExecuteMode.NORMAL:
            if not get_metadata:
                engine_creator = BasicQueryEngineCreator()
                # data_processor = BasicDataProcessor()
            else:
                #TODO: meter al fichero de configuracion los extractores
                engine_creator = RouterRetrieverQueryEngineCreator(metadata_config, KeywordsExtractors([KeywordsNLPExtractor(), KeywordsKeyBertExtractor()]))
                # data_processor = GetMetadataDataProcessor(metadata_config)
        
        data_processor = DataProcessor(
                full_config.app.general.data_folder_path,
                full_config.app.pre_retrieval.get_metadata,
                full_config.app.pre_retrieval.metadata_config,
                full_config.app.pre_retrieval.use_chunks,
                full_config.app.pre_retrieval.chunks_config
            )

        return RAGManager(processor, engine_creator, data_processor, full_config)

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
        st.session_state.rag_manager.init()

load_data()

def clean_message(self, text):
    # Eliminar contenido entre <think> y </think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Reemplazar URLs por localhost (por ejemplo, cambiar https://ejemplo.com/documento.pdf a http://localhost/documento.pdf)
    text = re.sub(r"https?://[^\s)]+", lambda match: match.group(0).replace(match.group(0).split('/')[2], "localhost"), text)

# load_data()
chat_engine = st.session_state.rag_manager.query_engine

if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(thinking_message):
            print(f"Pregunta: {prompt}")
            transformed_query = st.session_state.rag_manager.query_response_processor.transform(prompt)
            # question_with_prompt = f"{prompts[index]} Pregunta: {question}" 
            response = st.session_state.rag_manager.query_engine.query(transformed_query)  # Ejecutar consulta
            response_text = st.session_state.rag_manager.query_response_processor.process(response)
            st.write(response_text)
            print(f"Respuesta: {response_text}\n")

            # cleaned_content = clean_message(response.response)  # Aplicamos el filtro
            # st.write(cleaned_content)
            # print(cleaned_content)
            message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(message)  # Add response to message history