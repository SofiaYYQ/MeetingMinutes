from abc import ABC, abstractmethod
import re
from typing import Any, List
from llama_index.core.prompts.utils import format_string
from llama_index.core.schema import Document

# from config_loader.builder_register import StepFactory
from config_loader.models import ApplyFiltersActionStepModel, BaseStepModel, CheckTermsInTextActionStepModel, CompositeStepModel, FormatDocumentsActionStepModel, IfStepModel, LLMCallStepModel, MetadataConfig
from utils.utils import Utils

class Step(ABC):
    def __init__(self, global_context):
        self.global_context = global_context

    @abstractmethod
    def run(self):
        pass

class LLMCallStep(Step):
    def __init__(self, model:LLMCallStepModel, global_context, llm_call, json_llm_call):
        super().__init__(global_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call

    def run(self):
        selected_llm_call = self.json_llm_call if self.model.json_output else self.llm_call
        
        formatted_prompt = format_string(self.model.prompt, **self.global_context)
        return selected_llm_call(formatted_prompt)


class FormatDocumentsActionStep(Step):
    def __init__(self, model:FormatDocumentsActionStepModel, global_context: dict[str, Any], metadata_config:MetadataConfig):
        super().__init__(global_context)
        self.metadata_config = metadata_config
        self.model = model
        self.documents = self.global_context.get(self.model.inputs[0])

    def run(self):
        documents_str = []
        for document in self.documents:
            doc_filename = document.metadata["file_name"]
            metadata_str = []
            for k, v in document.metadata.items():
                if k != "file_name":
                    metadata_info = Utils.get_metadata_info(self.metadata_config, k)
                    if metadata_info.type == "list":
                        list_str = "\n".join([f"- {e}" for e in v])
                        metadata_str.append(f"{k}({Utils.get_metadata_info(self.metadata_config, k).description}):\n{list_str}")
                    else:
                        metadata_str.append(f"{k}({Utils.get_metadata_info(self.metadata_config, k).description}): {v}")
            
            metadata_str = "\n".join(metadata_str)
            document_str = f"<<<Documento {doc_filename}:\n {metadata_str}>>>"
            documents_str.append(document_str)

        return "\n".join(documents_str)
    

class CompositeStep(Step):
    def __init__(self, model:CompositeStepModel, global_context: dict[str, Any], llm_call, json_llm_call):
        super().__init__(global_context)
        self.model = model
        self.local_context = {}
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call
    
    def run(self):
        for s in self.model.steps:
            step = StepFactory.create(
                s, 
                global_context = self.global_context | self.local_context, 
                llm_call = self.llm_call, 
                json_llm_call = self.json_llm_call
            )
            output = step.run()
            self.local_context[s.output] = output

        if output == None:
            local_context_values = list(self.local_context.values())
            output = local_context_values[-1] if len(local_context_values) > 0 else None

        return output
    
class IfStep(Step):
    def __init__(self, model:IfStepModel, global_context: dict[str, Any], llm_call, json_llm_call):
        super().__init__(global_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call

    def run(self):
        condition = self.model.condition
        output = None
        isTrue = eval(condition)
        # if_false es opcional, entonces se puede dar que output devuelva None
        s = self.model.if_true if isTrue else self.model.if_false  

        if s:
            step = StepFactory.create(
                s, 
                global_context = self.global_context, 
                llm_call = self.llm_call, 
                json_llm_call = self.json_llm_call
            )
            output = step.run()

        return output

class ApplyFiltersActionStep(Step):
    def __init__(self, model:ApplyFiltersActionStepModel, global_context: dict[str, Any]):
        super().__init__(global_context)
        self.model = model
        self.documents = self.global_context.get(self.model.inputs[0])
        self.filters = self.group_filters()

    def group_filters(self):
        all_filters = self.model.inputs[1:]
        all_filters_objs = [self.global_context.get(f) for f in all_filters]
        return {k: v for d in all_filters_objs for k, v in d.items() if v is not None or v != "None"}

    def run(self):
        filtered = []
        unmatched_values = list(self.filters.values())
        for doc in self.documents:
            doc_matched = True

            for key, value in self.filters.items():
                metadata_value = str(doc.metadata.get(key, ""))

                if key == "fecha":
                    value_pattern = value.replace("%", ".*")
                    if not re.search(value_pattern, metadata_value):
                        doc_matched = False
                    else:
                        if value in unmatched_values:
                            unmatched_values.remove(value)
                else:
                    if value not in metadata_value:
                        doc_matched = False
                    else:
                        if value in unmatched_values:
                            unmatched_values.remove(value)
                        
            if doc_matched:
                filtered.append(doc)


        return filtered, unmatched_values

class CheckTermsInTextActionStep(Step):
    def __init__(self, model:CheckTermsInTextActionStepModel, global_context: dict[str, Any]):
        super().__init__(global_context)
        self.model = model
        self.terms = self.model.inputs[0]
        self.text = self.global_context.get(self.model.inputs[1])

    def run(self):
        query_terms = self.text.lower().split(" ")
        for term in query_terms:
            if term in self.terms:
                return True
        return False
    
class StepFactory:
    @staticmethod
    def create(model: BaseStepModel, **kwargs) -> Step:
        if isinstance(model, LLMCallStepModel):
            # from config_loader.steps import LLMCallStep
            return LLMCallStep(
                model, 
                global_context = kwargs.get("global_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
            )
        elif isinstance(model, FormatDocumentsActionStepModel):
            # from config_loader.steps import FormatDocumentsActionStep
            return FormatDocumentsActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                metadata_config = kwargs.get("metadata_config")
            )
        elif isinstance(model, CompositeStepModel):
            # from config_loader.steps import CompositeStep
            return CompositeStep(
                model, 
                global_context = kwargs.get("global_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
            )
        elif isinstance(model, ApplyFiltersActionStepModel):
            # from config_loader.steps import ApplyFiltersActionStep
            return ApplyFiltersActionStep(
                model, 
                global_context = kwargs.get("global_context")
            )
        elif isinstance(model, CheckTermsInTextActionStepModel):
            # from config_loader.steps import ApplyFiltersActionStep
            return CheckTermsInTextActionStep(
                model, 
                global_context = kwargs.get("global_context")
            )
        else:
            raise ValueError(f"No logic builder registered for model {type(model).__name__}")