import os

from abc import ABC, abstractmethod
from typing import List

from config_loader.models import EvaluationConfig, FullConfig
# from rag_manager import RAGManager
from evaluate.accuracy_evaluator import AccuracyEvaluator

# from executions.executions import ModeExecution
from logger_manager import LoggerMixin
from utils.utils import Utils
from utils.evaluation_mode_validator import EvaluationModeValidator
from utils.file_handler import FileHandler
from llama_index.core.workflow import Workflow
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts.utils import format_string
from llama_index.core.agent.workflow import ReActAgent

class WorkflowModeExecution(ABC, LoggerMixin):
    def __init__(self, workflow: Workflow):
        super().__init__()
        self.workflow = workflow
        
    @abstractmethod
    def run(self):
        pass

class WorkflowEvaluationModeExecution(WorkflowModeExecution):
    def __init__(self, workflow: Workflow, evaluation_config: EvaluationConfig, documents:List[Document]):
        super().__init__(workflow)
        self.evaluation_config = evaluation_config
        self.validator = EvaluationModeValidator()
        self.documents = documents

    async def run(self):
        await self._process_questions()

    async def _process_question(self, query)-> str:
        response = await self.workflow.run(
            documents=self.documents,
            query=query
        )
        return response
    
    async def _process_questions(self):
        questions_file_path = self.evaluation_config.questions_file_path
        prompts_file_path = self.evaluation_config.prompts_file_path
        answers_file_path = self.evaluation_config.answers_file_path
        results_folder_path = self.evaluation_config.results_folder_path
        reports_folder_path = self.evaluation_config.reports_folder_path

        questions = FileHandler.read_from_txt(questions_file_path)
        prompts = FileHandler.read_from_txt(prompts_file_path)
        answers = FileHandler.read_from_txt(answers_file_path)

        responses = []

        for index, question in enumerate(questions):
            self.logger.info(f"{index + 1}. Pregunta: {question}")
            
            response_text = await self._process_question(question)

            self.logger.info(f"Respuesta: {response_text}\n")
            responses.append(response_text) 

        # Get the questions file name (without .txt extension and parent folders)
        questions_file_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    
        # filename = Utils.get_analysis_output_name(questions_file_name, self.full_config)
        filename = Utils.get_testing_analysis_output_name(questions_file_name)

        FileHandler.write_to_txt(f"{results_folder_path}/{filename}.txt", responses)

        formatted_real_responses = self.validator.get_formatted_answers(responses, prompts)
        AccuracyEvaluator.get_accuracy_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses, formatted_real_responses)

        # AccuracyEvaluator.get_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses)

class BaselineEvaluationModeExecution(LoggerMixin):
    def __init__(self, llm: Ollama, evaluation_config: EvaluationConfig, documents:List[Document]):
        super().__init__()
        self.llm = llm
        self.evaluation_config = evaluation_config
        self.validator = EvaluationModeValidator()
        self.documents = documents
        self.prompt = (
            "Información contenida en los pdfs está a continuación:\n"
            "{context_str}"
            "Dada la información contenida en los pdfs y sin conocimientos previos.\n"
            "Responde a la consulta.\n"
            "Consulta: {query_str}\n"
        )

    def run(self):
        self._process_questions()

    def _process_question(self, query)-> str:
        docs_with_metadata = [f"{d.metadata}\n{d.text}" for d in self.documents]
        context_str = "\n\n".join([f"Fichero {i+1}:\n {f}" for i, f in enumerate(docs_with_metadata)])
        response = self.llm.complete(format_string(self.prompt, context_str=context_str, query_str=query))
        response_text = response.text
        return response_text
    
    def _process_questions(self):
        questions_file_path = self.evaluation_config.questions_file_path
        prompts_file_path = self.evaluation_config.prompts_file_path
        answers_file_path = self.evaluation_config.answers_file_path
        results_folder_path = self.evaluation_config.results_folder_path
        reports_folder_path = self.evaluation_config.reports_folder_path

        questions = FileHandler.read_from_txt(questions_file_path)
        prompts = FileHandler.read_from_txt(prompts_file_path)
        answers = FileHandler.read_from_txt(answers_file_path)

        responses = []

        for index, question in enumerate(questions):
            self.logger.info(f"{index + 1}. Pregunta: {question}")
            
            response_text = self._process_question(question)

            self.logger.info(f"Respuesta: {response_text}\n")
            responses.append(response_text) 

        # Get the questions file name (without .txt extension and parent folders)
        questions_file_name = os.path.splitext(os.path.basename(questions_file_path))[0]
        model_name = self.llm.model.replace(":", "_")
        # filename = Utils.get_analysis_output_name(questions_file_name, self.full_config)
        filename_custom_part = f"baseline_{model_name}"
        filename = Utils.get_custom_analysis_output_name(questions_file_name, filename_custom_part)

        FileHandler.write_to_txt(f"{results_folder_path}/{filename}.txt", responses)

        formatted_real_responses = self.validator.get_formatted_answers(responses, prompts)
        AccuracyEvaluator.get_accuracy_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses, formatted_real_responses)

        # AccuracyEvaluator.get_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses)


class ReActAgentEvaluationModeExecution(LoggerMixin):
    def __init__(self, llm: Ollama, evaluation_config: EvaluationConfig, documents:List[Document]):
        super().__init__()
        self.llm = llm
        self.evaluation_config = evaluation_config
        self.validator = EvaluationModeValidator()
        self.documents = documents
        self.prompt = (
            "Información contenida en los pdfs está a continuación:\n"
            "{context_str}"
            "Dada la información contenida en los pdfs y sin conocimientos previos.\n"
            "Responde a la consulta.\n"
            "Consulta: {query_str}\n"
        )
        self.agent = ReActAgent(
            tools=[],
            llm=self.llm, 
            verbose=True,
            max_iterations=3
        )


    async def run(self):
        await self._process_questions()

    async def _process_question(self, query)-> str:
        docs_with_metadata = [f"{d.metadata}\n{d.text}" for d in self.documents]
        context_str = "\n\n".join([f"Fichero {i+1}:\n {f}" for i, f in enumerate(docs_with_metadata)])

        formatted_prompt = format_string(self.prompt, context_str=context_str, query_str=query)
        response = await self.agent.run(formatted_prompt, timeout=1200)
        response_text = str(response)
        return response_text
    
    async def _process_questions(self):
        questions_file_path = self.evaluation_config.questions_file_path
        prompts_file_path = self.evaluation_config.prompts_file_path
        answers_file_path = self.evaluation_config.answers_file_path
        results_folder_path = self.evaluation_config.results_folder_path
        reports_folder_path = self.evaluation_config.reports_folder_path

        questions = FileHandler.read_from_txt(questions_file_path)
        prompts = FileHandler.read_from_txt(prompts_file_path)
        answers = FileHandler.read_from_txt(answers_file_path)

        responses = []

        for index, question in enumerate(questions):
            self.logger.info(f"{index + 1}. Pregunta: {question}")
            
            response_text = await self._process_question(question)

            self.logger.info(f"Respuesta: {response_text}\n")
            responses.append(response_text) 

        # Get the questions file name (without .txt extension and parent folders)
        questions_file_name = os.path.splitext(os.path.basename(questions_file_path))[0]
        model_name = self.llm.model.replace(":", "_")
        # filename = Utils.get_analysis_output_name(questions_file_name, self.full_config)
        filename_custom_part = f"ReAct_{model_name}"
        filename = Utils.get_custom_analysis_output_name(questions_file_name, filename_custom_part)

        FileHandler.write_to_txt(f"{results_folder_path}/{filename}.txt", responses)

        formatted_real_responses = self.validator.get_formatted_answers(responses, prompts)
        AccuracyEvaluator.get_accuracy_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses, formatted_real_responses)

        # AccuracyEvaluator.get_results(f"{reports_folder_path}/{filename}.csv", questions, answers, responses)