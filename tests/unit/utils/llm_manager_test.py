

from config_loader.models import LLMConfig
from utils.llm_manager import LLMManager

from unittest.mock import MagicMock, patch


class TestLLMManager:
    @patch('utils.llm_manager.Ollama')
    @patch('utils.llm_manager.OllamaEmbedding')
    @patch('utils.llm_manager.Settings')
    def test_init(self, mock_settings, mock_embed,mock_llm):
        model_name = "llama3.2"
        embedding_model_name = "snowflake-arctic-embed2"
        request_timeout = 600.0
        temperature = 0.3
        llm_config = LLMConfig(
            model_name = model_name,
            embedding_model_name = embedding_model_name,
            request_timeout = request_timeout,
            temperature = temperature
        )
        
        LLMManager().init(llm_config)


        mock_ollama_instance = MagicMock()
        mock_llm.return_value = mock_ollama_instance

        mock_embed_instance = MagicMock()
        mock_embed.return_value = mock_embed_instance

        mock_llm.assert_called_once_with(model=model_name, temperature=temperature, request_timeout=request_timeout)
        mock_embed.assert_called_once_with(model_name=embedding_model_name)

        assert mock_settings.llm is not None
        assert mock_settings.embed_model is not None

    @patch('utils.llm_manager.Ollama')
    def test_create_json_output_llm(self, mock_llm):
        model_name = "gemma3:12b"
        embedding_model_name = "snowflake-arctic-embed2"
        request_timeout = 600.0
        temperature = 0.0
        json_mode = True

        llm_config = LLMConfig(
            model_name = model_name,
            embedding_model_name = embedding_model_name,
            request_timeout = request_timeout
        )

        LLMManager.create_json_output_llm_by_config(llm_config)

        mock_ollama_instance = MagicMock()
        mock_llm.return_value = mock_ollama_instance
        mock_llm.assert_called_once_with(model=model_name, temperature=temperature, request_timeout=request_timeout, json_mode=json_mode)
        
