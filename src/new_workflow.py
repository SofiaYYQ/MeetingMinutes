import json
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
# import asyncio
from llama_index.core.schema import Document
# from llama_index.core.prompts.utils import format_string
# from config_loader.builders import KeywordsExtractorCreator
# from config_loader.models import KeyBertKeywordsExtractorConfig, KeywordsExtractorsConfig, NLPKeywordsExtractorConfig
# from data_processors.static_data_processor import StaticDataProcessor
# from config_loader.builder_register import StepFactory
# from config_loader.builder_register import step_logic_factory
from config_loader.models import BaseStepModel, MetadataConfig
from config_loader.steps import StepFactory
from logger_manager import LoggerManager

# from llama_index.core import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# from query_engines.key_extractors import IKeywordsExtractor
from utils.llm_call_manager import LLMCallManager
# from utils.sql_parser import SQLParser
import re

from utils.utils import Utils
# from llama_index.core.llms.llm import LLM

# pasar a utils
def get_metadata_info(metadata_config:MetadataConfig, metadata_name:str|None):
    for metadata in metadata_config.metadata_info:
        if metadata.name == metadata_name:
            return metadata
    return None

class DocumentsBasedQAFlowExecutor():
    def __init__(self, llm:Ollama, llm_json_output:Ollama, metadata_config:MetadataConfig, documents: List[Document], workflow: List[BaseStepModel]) -> None:
        self.logger = LoggerManager.get_logger(name=self.__class__.__name__)
        self.llm = llm
        self.llm_json_output = llm_json_output
        self.metadata_config = metadata_config
        self.documents = documents
        self.step_results = []
        self.workflow = workflow
        self.context = {}
        self.context["documents"] = documents
    

    def run(self, query:str)->str:
        self.context["query"] = query

        
        steps_by_id = {s.id: s for s in self.workflow}
        step_ids = list(steps_by_id.keys())
        current_step_index = 0
        output = None

        while current_step_index < len(step_ids):
            step_id = step_ids[current_step_index]
            s = steps_by_id[step_id]

            step = StepFactory.create(
                s,
                global_context=self.context,
                llm_call=self.get_llm_output,
                json_llm_call=self.get_valid_json_output,
                metadata_config=self.metadata_config
            )

            output = step.run()
            
            if isinstance(output, dict) and "go_to" in output:
                target_id = output["go_to"]
                if target_id in steps_by_id:
                    current_step_index = step_ids.index(target_id)
                    continue
                else:
                    raise ValueError(f"Destination step '{target_id}' not found.")
                
            self.context[s.output] = output
            current_step_index += 1

        if output is None:
            context_values = list(self.context.values())
            output = context_values[-1] if context_values else "No output."

        self.reset()
        return output

        # for s in self.workflow:
        #     # step = step_logic_factory.create(s, global_context=self.context, llm_call = self.llm.complete, json_llm_call = self.get_valid_json_output)
        #     step = StepFactory.create(
        #         s, 
        #         global_context=self.context, 
        #         llm_call = self.get_llm_output, 
        #         json_llm_call = self.get_valid_json_output,
        #         metadata_config = self.metadata_config
        #     )
        #     output = step.run()
        #     self.context[s.output] = output

        # if output == None:
        #     context_values = list(self.context.values())
        #     output = context_values[-1] if len(context_values) > 0 else "No output."
    
        # self.reset()

        # return output
    
    def reset(self):
        self.step_results = {}
        self.context = {}
        self.context["documents"] = self.documents

    def get_llm_output(self, prompt):
        response = self.llm.complete(prompt)
        return response.text
    
    def process_complete_response(self, complete):
        return LLMCallManager.process_complete_response(self.llm.model, complete)
    
#     def init_step_results(self):
#         self.step_results = []
#     @step
#     async def filter_documents(self, ev: StartEvent) -> GetFinalResponseEvent:
#         self.init_step_results()
#         query = ev.query
#         documents = ev.documents

#         extract_name_prompt = (
# f"""
# Extrae el nombre completo (nombre y apellidos) de la persona mencionada en la siguiente consulta. Si no se menciona ninguna persona, devuelve "None".

# Devuelve el resultado en formato JSON con la clave "persona", siguiendo exactamente esta estructura:

# {{"persona": "nombre y apellidos de la persona"}}
# o
# {{"persona": "None"}}

# CONSULTA: {query}
# """)
        
#         extract_date_prompt = (
# f"""

# Extrae la fecha mencionada en la siguiente consulta en formato DD/MM/AAAA. Si no se menciona ninguna fecha, devuelve "None".

# Si la fecha está incompleta, utiliza "%" para el día, el mes o el año faltante. Por ejemplo:
# - Si solo se menciona "junio de 2023", devuelve: {{"fecha": "%/06/2023"}}
# - Si solo se menciona "2022", devuelve: {{"fecha": "%/%/2022"}}

# Devuelve el resultado en formato JSON con la clave "fecha", siguiendo exactamente una de estas dos estructuras:

# {{"fecha": "DD/MM/AAAA"}}
# o
# {{"fecha": "None"}}

# CONSULTA: {query}
# """)
#         filters = {}
#         name_response = self.get_valid_json_output(extract_name_prompt, ["persona"])
#         if name_response["persona"] != "None":
#             filters["lista_asistentes"] = name_response["persona"]
#         self.logger.info(f"Output. Extract name: {str(name_response)}")
        

#         date_response = self.get_valid_json_output(extract_date_prompt, ["fecha"])
#         if date_response["fecha"] != "None":
#             filters["fecha"] = date_response["fecha"]
#         self.logger.info(f"Output. Extract date: {str(date_response)}")

#         # Step 2: Filter documents if both name and date exist
#         step_result = {
#             "name": "Filtrado de documentos",
#             "description": "Se identifican y excluyen los documentos que no cumplen con los criterios establecidos en la consulta.",
#         }
#         if filters:
#             filtered_docs, unmatched_values = self.filter_documents_by_metadata(documents, filters)
#             formatted_no_relevant_docs = []

#             for d in documents:
#                 if d not in filtered_docs:
#                     nombre = d.metadata["file_name"]
#                     fecha = d.metadata["fecha"]
#                     formatted_no_relevant_docs.append(f"Documento {nombre} ({fecha})")
#             no_relevant_docs_str = ", ".join(formatted_no_relevant_docs)

#             if len(filtered_docs) == 0:
#                 # filters_values_str = ", ".join([v for k, v in filters.items()])
#                 unmatched_values_str = ", ".join(unmatched_values)
#                 step_result["resultado"] = (
#                     f"A partir de la consulta '{query}', se han identificado los siguientes filtros aplicables: {filters}.\n"
#                     f"Tras aplicar estos filtros, se ha determinado que ninguno de los documentos disponibles cumplen los criterios.\n"
#                     f"Estos son los VALROES que no se mencionan en ningún documento: [{unmatched_values_str}].\n"
#                     "Esto puede deberse a varias razones:\n"
#                     "- Si no se menciona el nombre, es posible que la persona no figure en los registros o que su nombre esté mal escrito.\n"
#                     "- Si no se menciona la fecha, es probable que no existan reuniones registradas en esa fecha específica.\n"
#                     "- Si no se menciona ni el nombre ni la fecha, probablemente que no existe ni la persona ni la reunión en esa fecha. \n"
#                     "En caso de que la lista de VALORES está vacía, significa que la persona y la reunión sí existen, pero la persona no asistió a la reunión de la fecha indicada.\n"
#                     f"Por lo tanto, todos los documentos han sido descartados. Documentos excluidos: {no_relevant_docs_str}."
#                 )

#                 self.step_results.append(step_result)
#                 return GetFinalResponseEvent(query=query)
#             else:
#                 # Elimina la última coma y espacio si es necesario
#                 step_result["resultado"] = (
#                     f"A partir de la consulta '{query}', se han identificado los siguientes filtros aplicables: {filters}. "
#                     "Se han revisado todos los documentos disponibles y se han excluido aquellos que no coinciden con los filtros especificados."
#                     f"Los documentos que no cumplen con estos criterios son los siguientes: {no_relevant_docs_str}"
#                 )
#                 self.step_results.append(step_result)
#         else:
#             filtered_docs = documents
#             step_result["resultado"] = (
#                 f"A partir de la consulta '{query}', no se han extraído ningún filtro. "
#                 "Por lo tanto no se excluyen ningún documento."
#             )
#             self.step_results.append(step_result)
#         relevant_docs_str = ", ".join([d.metadata["file_name"] for d in filtered_docs])
#         self.logger.info(f"Filtered Documents: {relevant_docs_str}")
        
            
#         # # Step 3: Check for global query and 
#         is_global_query = self.is_global_query(query)
#         is_comparative_query = self.is_comparative_query(query)

#         step_result = {
#             "name": "Análisis del tipo de consulta",
#             "description": (
#                 "Se clasifica la consulta como global o individual. "
#                 "Una consulta global (normal o comparativa) requiere obtener información de múltiples reuniones, "
#                 "mientras que una consulta individual se centra en una única reunión."
#             ),
#         }
#         #Step 4: Transform query if it is global
#         if not is_global_query:
#             query_to_analyze = query
#             step_result["resultado"] = f"""Según el análisis, la consulta '{query}' es de tipo individual. Por ello, no hay necesidad de transformarla en una subconsulta."""
#         else:
#             if is_comparative_query:
#                 step_result["resultado"] = (
#                     f"Según el análisis, la consulta '{query}' es de tipo global. En este caso, no se tranforma en una subconsulta."
#                 )
#             else:
#                 self.logger.info(f"This is a global query: {query}")
#                 query_to_analyze = self.trasform_to_sub_query(query)
#                 step_result["resultado"] = (
#                 f"Según el análisis, la consulta '{query}' es de tipo global. Por ello, se obtiene una subconsulta para preguntar a cada reunión: {query_to_analyze}"
#             )

#         self.step_results.append(step_result)

#         # Step 5: Return result
#         step_result = {
#             "name": "Recopilación de información",
#             "description": "Se extrae información relevante de cada documento previamente filtrado mediante la consulta, con el objetivo de obtener los datos necesarios para su posterior análisis.",
#         }


#         if not is_comparative_query:
#             evidences = []
#             for d in filtered_docs:
#                 response_str = self.get_evidence(d, query_to_analyze) # objeto jsons
#                 for time in range(5):
#                     isTrue = self.evaluate(d, response_str)
#                     if isTrue:
#                         break
#                     response_str = self.get_evidence(d, query_to_analyze)
                
#                 evidences.append((d, response_str))
#             info_str = self.format_evidences(evidences)

#         else:
#             info_str = self.format_documents(filtered_docs)

        
#         step_result["resultado"] = f"Se ha recopilado la siguiente información:\n {info_str}"

#         self.step_results.append(step_result)
#         return GetFinalResponseEvent(query=query)

#     @step
#     def get_final_response(self, ev:GetFinalResponseEvent)->StopEvent:
#         query = ev.query
#         formatted_steps = "".join(
#             f"Paso {i}: {step['name']}\nDescripción: {step['description']}\nResultado: {step['resultado']}\n\n"
#             for i, step in enumerate(self.step_results, 1)
#         )

        
#         prompt = f"""
# Genera una respuesta final directa a la consulta del usuario, utilizando únicamente la información contenida en los pasos que se presentan a continuación. 
# No expliques los pasos ni los filtros aplicados. Limítate a responder a la consulta como si ya hubieras realizado todo el análisis necesario.

# Pasos realizados anteriormente:
# {formatted_steps}

# Consulta del usuario: {query}

# """

#         self.logger.info(f"Prompt. Final Response Generator: {prompt}")
#         response = self.llm.complete(prompt)
#         response_str = self.process_complete_response(response)
#         self.logger.info(f"Output. Final Response Generator: {response_str}")

#         return StopEvent(result=response_str)

    # def filter_documents_by_metadata(
    #     self,
    #     documents: List[Document],
    #     filters: Dict[str, Union[str, List[str]]]
    # ) -> Tuple[List[Document], List[str]]:
    #     filtered = []
    #     unmatched_values = list(filters.values())
    #     for doc in documents:
    #         doc_matched = True

    #         for key, value in filters.items():
    #             metadata_value = str(doc.metadata.get(key, ""))

    #             if key == "fecha":
    #                 value_pattern = value.replace("%", ".*")
    #                 if not re.search(value_pattern, metadata_value):
    #                     doc_matched = False
    #                 else:
    #                     if value in unmatched_values:
    #                         unmatched_values.remove(value)
    #             else:
    #                 if value not in metadata_value:
    #                     doc_matched = False
    #                 else:
    #                     if value in unmatched_values:
    #                         unmatched_values.remove(value)
                        
    #         if doc_matched:
    #             filtered.append(doc)


    #     return filtered, unmatched_values
    
    def get_valid_json_output(self, prompt, keys=None):
        if keys == None:
            keys = Utils.extract_json_keys_from_text(prompt)
        while True:
            try:
                response = self.llm_json_output.complete(prompt)
                response_str = self.process_complete_response(response)
                data = json.loads(response_str)
                
                if keys:
                    if all([key in data for key in keys]):
                        self.logger.info("Valid JSON: %s", data)
                        return data
                else:
                    self.logger.info("Valid JSON: %s", data)
                    return data
            except json.JSONDecodeError:
                self.logger.info("Invalid JSON. Try it again.")

    


#     def is_global_query(self, query:str):
#         results = []
#         results.append("reuniones" in query.lower())

#         results.append(self.is_comparative_query(query))
        
#         query_terms = query.lower().split(" ")
#         results.append("reunión" not in query_terms)
#         return any(results)

#     def is_comparative_query(self, query:str):
#         comparison_terms = [
#             # Evaluativos
#             "mejor", "mejores", "peor", "peores", "superior", "superiores", "inferior", "inferiores",
#             "mayor", "mayores", "menor", "menores", "más", "menos"
#         ]

#         query_terms = query.lower().split(" ")
#         for term in query_terms:
#             if term in comparison_terms:
#                 return True
            
#         return False
    
#     def trasform_to_sub_query(self, query:str):
#         prompt = (f"""
#         Transforma preguntas globales que requieren revisar múltiples documentos (como actas de reuniones) en preguntas individuales que puedan aplicarse a cada documento por separado.
#         Pasos:
#         1.	Identificar el sujeto y la acción de la oración:
#         2.	Convierte la pregunta global en una individual:
#         ¿[Acción] [Sujeto] [Por el complemento que se pregunta si existe, sino este documento]?
#         Ejemplo
#         Pregunta global: ¿En cuántos informes médicos se menciona a Laura Sánchez?
#         Pregunta individual: ¿Es mencionada Laura Sánchez en este informe médico?

#         Ejemplo
#         Pregunta global: ¿Cuántas veces aparece el nombre de Javier Torres?
#         Pregunta individual: ¿Aparece el nombre de Javier Torres en este documento?
                    
#         Transforma ahora la siguiente pregunta global en una pregunta individual aplicable a un solo documento. 
#         Pregunta global: {query}

#         Responde en formato json con dos claves "pregunta_global", "pregunta_individual":
#         {{
#             "pregunta_global":"pregunta original",
#             "pregunta_individual":"pregunta transformada"
#         }}
#         """)
#         self.logger.info(f"Prompt. Transform to a individual question: {prompt}")
#         data = self.get_valid_json_output(prompt)
#         self.logger.info(f"Output. Transform to a individual question: {str(data)}")

#         return data["pregunta_individual"]
    
#     def get_evidence(self, document:Document, query:str):
#         doc_filename = document.metadata["file_name"]
#         metadata_str = []
#         for k, v in document.metadata.items():
#             if k != "file_name":
#                 metadata_info = get_metadata_info(k)
#                 if metadata_info.type == "list":
#                     list_str = "\n".join([f"- {e}" for e in v])
#                     metadata_str.append(f"{k}({get_metadata_info(k).description}):\n{list_str}")
#                 else:
#                     metadata_str.append(f"{k}({get_metadata_info(k).description}): {v}")

#         metadata_str = "\n".join(metadata_str)
#         # TODO: Probar pasar directament el json  a ver si funciona
#         document_str = f"<<<Documento {doc_filename}:\n {metadata_str}>>>"
        
#         prompt = (
#         f"""          
#         Contexto:
#         {document_str}

#         Analiza el documento proporcionado y utiliza esa información para responder con precisión a la siguiente pregunta.

#         Instrucciones:

#         1. Identifica los datos del documento de donde proviene la información, incluyendo el nombre del archivo y la fecha de la reunión.

#         2. Redacta una respuesta clara, completa y contextualizada, que:
#         - Sea directa, bien fundamentada y autónoma.

#         - Incluya explícitamente el sujeto, la acción y los datos de identificación del documento (como el nombre del archivo, fecha de reunión) dentro del cuerpo de la respuesta, para que quede claro a qué documento se refiere.

#         - Evite respuestas fragmentadas (por ejemplo, no uses "Sí, llegó", usa "Sí, Ángel llegó el viernes, según el documento [nombre del documento y otros datos de identificacón del documento]").
        
#         3. Incluye evidencia textual relevante extraída del documento para respaldar tu respuesta. Esta puede ser una cita directa o un resumen claro y preciso.


#         Formato de salida (JSON):
#         {{
#         "pregunta": "(escribe aquí la pregunta original)",
#         "respuesta": "(respuesta clara, completa, con contexto y referencia explícita al documento)",
#         "evidencia": "(cita textual o resumen de la evidencia encontrada en el documento)"
#         }}
#         Pregunta: {query}
#         """
#         )
#         self.logger.info(f"Prompt. Get response by document: {prompt}")
#         data = self.get_valid_json_output(prompt)
#         self.logger.info(f"Output. Get response by document: {str(data)}")

#         answer = data["respuesta"]
#         evidence = data["evidencia"]
#         return f"{answer} (Evidencia textual: {evidence})"
    
#     def format_evidences(self, evidences):
#         evidences_str = ""
#         for i, e in enumerate(evidences):
#             document = e[0]
#             doc_filename = document.metadata["file_name"]
#             doc_id = document.metadata["fecha"]
#             response = e[1]
            
#             evidences_str += f"- Documento {doc_filename}({doc_id}): {response}\n"

#         return evidences_str
    
#     def format_documents(self, documents):
#         documents_str = []
#         for document in documents:
#             doc_filename = document.metadata["file_name"]
#             metadata_str = []
#             for k, v in document.metadata.items():
#                 if k != "file_name":
#                     metadata_info = get_metadata_info(k)
#                     if metadata_info.type == "list":
#                         list_str = "\n".join([f"- {e}" for e in v])
#                         metadata_str.append(f"{k}({get_metadata_info(k).description}):\n{list_str}")
#                     else:
#                         metadata_str.append(f"{k}({get_metadata_info(k).description}): {v}")
            
#             metadata_str = "\n".join(metadata_str)
#             document_str = f"<<<Documento {doc_filename}:\n {metadata_str}>>>"
#             documents_str.append(document_str)

#         return documents_str
    

#     def evaluate(self, document:Document, answer):
#         doc_filename = document.metadata["file_name"]
#         metadata_str = []
#         for k, v in document.metadata.items():
#             if k != "file_name":
#                 metadata_info = get_metadata_info(k)
#                 if metadata_info.type == "list":
#                     list_str = "\n".join([f"- {e}" for e in v])
#                     metadata_str.append(f"{k}({get_metadata_info(k).description}):\n{list_str}")
#                 else:
#                     metadata_str.append(f"{k}({get_metadata_info(k).description}): {v}")
#         # metadata_str = '\n'.join(f'{k} ({get_metadata_info(k).description}): {v}' for k, v in d.metadata.items() if k != "file_name")
#         metadata_str = "\n".join(metadata_str)
#         # TODO: Probar pasar directament el json  a ver si funciona
#         document_str = f"<<<Documento {doc_filename}:\n {metadata_str}>>>"

#         prompt = (f""" 
# Evalúa si la respuesta está respaldada por el contenido del documento proporcionado. Indica si la información mencionada en la respuesta se encuentra explícitamente en el texto o si ha sido inferida o inventada.
                  
# Documento: 
# {document_str}

# Respuesta: {answer}

# Responde en formato json con dos claves: "evaluation" y "justification" de tipo string
# {{
#     "evaluation":"Elegir una de las tres opciones: [Explícita/Inferida/Inventada]",
#     "justification":"Solo si es necesario"
# }}
# """)
#         self.logger.info(f"Prompt. Response Evaluator: {prompt}")
#         data = self.get_valid_json_output(prompt, ["evaluation", "justification"])
#         self.logger.info(f"Output. Response Evaluator: {str(data)}")

#         return not data["evaluation"].lower().startswith("inventada")