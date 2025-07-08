from abc import ABC, abstractmethod
import re
from typing import Any, Callable
from llama_index.core.prompts.utils import format_string

from config_loader.models import AddToMemoryActionStepModel, ApplyFiltersActionStepModel, BaseStepModel, CheckTermsInTextActionStepModel, CompositeStepModel, EvaluateActionStepModel, ForEachStepModel, FormatDocumentsActionStepModel, FormatListActionStepModel, FormatMemoryActionStepModel, GoToStepModel, IfStepModel, LLMCallStepModel, MetadataConfig, SetVariableStepModel, FormatDocumentActionStepModel
from logger_manager import LoggerMixin
from utils.utils import Utils

class Step(ABC, LoggerMixin):
    def __init__(self, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], ):
        super().__init__()
        self.global_context = global_context
        self.add_to_context = add_to_context

    @abstractmethod
    def run(self):
        pass

class LLMCallStep(Step):
    def __init__(self, model:LLMCallStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], llm_call, json_llm_call):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call

    def run(self):
        selected_llm_call = self.json_llm_call if self.model.json_output else self.llm_call
        
        formatted_prompt = format_string(self.model.prompt, **self.global_context)
        
        self.logger.info(f"Prompt. Step \"{self.model.id}\": {formatted_prompt}")
        result = selected_llm_call(formatted_prompt)
        self.logger.info(f"Output. Step \"{self.model.id}\": {str(result)}")

        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class FormatDocumentsActionStep(Step):
    def __init__(self, model:FormatDocumentsActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], metadata_config:MetadataConfig):
        super().__init__(global_context, add_to_context)
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

        
        result = "\n".join(documents_str)
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result
        
class FormatDocumentActionStep(Step):
    def __init__(self, model:FormatDocumentActionStepModel, global_context, add_to_context:Callable[[str, Any], None], metadata_config:MetadataConfig):
        super().__init__(global_context, add_to_context)
        self.metadata_config = metadata_config
        self.model = model
        self.document = self.global_context.get(self.model.inputs[0])

    def run(self):
        doc_filename = self.document.metadata["file_name"]
        metadata_str = []
        for k, v in self.document.metadata.items():
            if k != "file_name":
                metadata_info = Utils.get_metadata_info(self.metadata_config, k)
                if metadata_info.type == "list":
                    list_str = "\n".join([f"- {e}" for e in v])
                    metadata_str.append(f"{k}({Utils.get_metadata_info(self.metadata_config, k).description}):\n{list_str}")
                else:
                    metadata_str.append(f"{k}({Utils.get_metadata_info(self.metadata_config, k).description}): {v}")
        
        metadata_str = "\n".join(metadata_str)
        document_str = f"<<<Documento {doc_filename}:\n {metadata_str}>>>"

        result = document_str
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result
    
class CompositeStep(Step):
    def __init__(
        self, 
        model:CompositeStepModel, 
        global_context: dict[str, Any], 
        add_to_context:Callable[[str, Any], None], 
        llm_call, 
        json_llm_call, 
        metadata_config:MetadataConfig, 
        add_to_memory, 
        format_memory
    ):
        super().__init__(global_context, add_to_context)
        self.model = model
        # self.local_context = {}
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call
        self.metadata_config = metadata_config
        self.add_to_memory = add_to_memory
        self.format_memory = format_memory

    def run(self):
        for s in self.model.steps:
            step = StepFactory.create(
                s, 
                # global_context = self.global_context | self.local_context, 
                global_context = self.global_context,
                add_to_context = self.add_to_context,
                llm_call = self.llm_call, 
                json_llm_call = self.json_llm_call,
                metadata_config = self.metadata_config,
                add_to_memory = self.add_to_memory,
                format_memory = self.format_memory
            )
            output = step.run()

            if isinstance(output, dict) and "go_to" in output:
                return output
            
        return output
    
class IfStep(Step):
    def __init__(self, model:IfStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], llm_call, json_llm_call, metadata_config, add_to_memory, format_memory):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call
        self.metadata_config = metadata_config
        self.add_to_memory = add_to_memory
        self.format_memory = format_memory

    def run(self):
        condition = self.model.condition
        output = None
        isTrue = self.evaluate_condition(condition, self.global_context)
        s = self.model.if_true if isTrue else self.model.if_false  

        if s:
            step = StepFactory.create(
                s, 
                global_context = self.global_context,
                add_to_context = self.add_to_context,
                llm_call = self.llm_call, 
                json_llm_call = self.json_llm_call,
                metadata_config = self.metadata_config,
                add_to_memory = self.add_to_memory,
                format_memory = self.format_memory
            )
            output = step.run()

            if isinstance(output, dict) and "go_to" in output:
                return output
    
        return output

    def evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        try:
            return eval(condition, {}, context)
        except Exception as e:
            self.logger.error(f"Error evaluating if condition in the step \"{self.model.id}\": {e}")
            return False

class ApplyFiltersActionStep(Step):
    def __init__(self, model:ApplyFiltersActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None]):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.documents = self.global_context.get(self.model.inputs[0])
        self.filters = self.group_filters()

    def group_filters(self):
        all_filters = self.model.inputs[1:]
        all_filters_objs = [self.global_context.get(f) for f in all_filters]

        combined_attrs = {}

        for obj in all_filters_objs:
            for k, v in vars(obj).items():
                if v is not None and v != "None":
                    combined_attrs[k] = v

        return combined_attrs

    def run(self):
        filtered = []
        unmatched_values = list(self.filters.values())
        discarded = []
        for doc in self.documents:
            doc_matched = True

            for key, value in self.filters.items():
                metadata_value = str(doc.metadata.get(key, ""))
                if metadata_value != "":
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
            else:
                discarded.append(str(doc.metadata.get("file_name", "")))

        result = [filtered, unmatched_values, self.filters, discarded]
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class CheckTermsInTextActionStep(Step):
    def __init__(self, model:CheckTermsInTextActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None]):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.terms = self.model.inputs[0]
        self.text = self.global_context.get(self.model.inputs[1])

    def run(self):
        result = False
        query_terms = self.text.lower().split(" ")
        for term in query_terms:
            if term in self.terms:
                result = True
        
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class GoToStep(Step):
    def __init__(self, model:GoToStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None]):
        super().__init__(global_context, add_to_context)
        self.model = model

    def run(self):
        return {"go_to": self.model.target_id}

class SetVariableStep(Step):
    def __init__(self, model:SetVariableStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None]):
        super().__init__(global_context, add_to_context)
        self.model = model

    def run(self):
        result = self.evaluate(self.model.source, self.global_context)
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result
    
    def evaluate(self, expr: str, context: dict[str, Any]) -> bool:
        try:
            return eval(expr, {}, context)
        except Exception as e:
            self.logger.error(f"Error evaluating if condition in the step \"{self.model.id}\": {e}")
            return None

class ForEachStep(Step):
    def __init__(self, model:ForEachStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], llm_call, json_llm_call, metadata_config: MetadataConfig, add_to_memory, format_memory):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call
        self.metadata_config = metadata_config
        self.add_to_memory = add_to_memory
        self.format_memory = format_memory

    def run(self):
        iterable_obj = self.global_context.get(self.model.iterate_obj)
        outputs = []
        for item in iterable_obj:
            self.add_to_context("item", item)
            step = StepFactory.create(
                self.model.step,
                global_context = self.global_context, 
                add_to_context = self.add_to_context,
                llm_call = self.llm_call, 
                json_llm_call = self.json_llm_call,
                metadata_config = self.metadata_config,
                add_to_memory = self.add_to_memory,
                format_memory = self.format_memory
            )
            output = step.run()
            if self.model.collected_field:
                outputs.append(self.global_context.get(self.model.collected_field))
            else:
                outputs.append(output)

        result = outputs
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class FormatListActionStep(Step):
    def __init__(self, model:FormatListActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None],):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.items = self.global_context.get(self.model.inputs[0])
        self.format_template = self.model.format_template
        self.separator = self.model.separator

    def run(self):
        formatted_items = [self.format_template.format(item=item) for item in self.items]
        result = self.separator.join(formatted_items)
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class AddToMemoryActionStep(Step):
    def __init__(self, model:AddToMemoryActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], add_to_memory):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.add_to_memory = add_to_memory

    def run(self):
        self.add_to_memory(
            self.model.name,
            self.model.description,
            format_string(self.model.result, **self.global_context),
        )

class FormatMemoryActionStep(Step):
    def __init__(self, model:AddToMemoryActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], format_memory):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.format_memory = format_memory

    def run(self):
        result = self.format_memory()
        output = self.model.output
        if output:
            self.add_to_context(output, result)
            return result

class EvaluateActionStep(Step):
    def __init__(self, model:EvaluateActionStepModel, global_context: dict[str, Any], add_to_context:Callable[[str, Any], None], llm_call, json_llm_call):
        super().__init__(global_context, add_to_context)
        self.model = model
        self.llm_call = llm_call
        self.json_llm_call = json_llm_call

    def run(self):
        selected_llm_call = self.json_llm_call if self.model.json_output else self.llm_call
        
        step = StepFactory.create(
            self.model.step,
            global_context = self.global_context, 
            add_to_context = self.add_to_context,
            llm_call = self.llm_call, 
            json_llm_call = self.json_llm_call
        )

        for i in range(self.model.max_intents):
            step.run()

            self.logger.info(f"Evaluation Intent {i+1}:")
            # formatted_prompt = format_string(self.model.prompt, **self.global_context)
            formatted_prompt = self.model.prompt.format(**self.global_context)
            
            self.logger.info(f"Prompt. Step \"{self.model.id}\": {formatted_prompt}")
            result = selected_llm_call(formatted_prompt)
            self.logger.info(f"Output. Step \"{self.model.id}\": {str(result)}")
            
            output = self.model.output
            if output:
                self.add_to_context(output, result)
        
            if self.evaluate_condition(self.model.condition):
                break

    def evaluate_condition(self, condition: str) -> bool:
        try:
            return eval(condition, {}, self.global_context)
        except Exception as e:
            self.logger.error(f"Error evaluating condition in the step \"{self.model.id}\": {e}")
            return False

    
class StepFactory:
    @staticmethod
    def create(model: BaseStepModel, **kwargs) -> Step:
        if isinstance(model, LLMCallStepModel):
            return LLMCallStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
            )
        elif isinstance(model, FormatDocumentsActionStepModel):
            return FormatDocumentsActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                metadata_config = kwargs.get("metadata_config")
            )
        elif isinstance(model, FormatDocumentActionStepModel):
            return FormatDocumentActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                metadata_config = kwargs.get("metadata_config")
            )
        elif isinstance(model, CompositeStepModel):
            return CompositeStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
                metadata_config = kwargs.get("metadata_config"),
                add_to_memory = kwargs.get("add_to_memory"),
                format_memory = kwargs.get("format_memory")
            )
        elif isinstance(model, ApplyFiltersActionStepModel):
            return ApplyFiltersActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
            )
        elif isinstance(model, CheckTermsInTextActionStepModel):
            return CheckTermsInTextActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
            )
        elif isinstance(model, IfStepModel):
            return IfStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
                metadata_config = kwargs.get("metadata_config"),
                add_to_memory = kwargs.get("add_to_memory"),
                format_memory = kwargs.get("format_memory")
            )
        elif isinstance(model, GoToStepModel):
            return GoToStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
            )
        elif isinstance(model, SetVariableStepModel):
            return SetVariableStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
            )
        elif isinstance(model, ForEachStepModel):
            return ForEachStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
                metadata_config = kwargs.get("metadata_config"),
                add_to_memory = kwargs.get("add_to_memory"),
                format_memory = kwargs.get("format_memory")
            )
        elif isinstance(model, FormatListActionStepModel):
            return FormatListActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
            )
        elif isinstance(model, AddToMemoryActionStepModel):
            return AddToMemoryActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                add_to_memory = kwargs.get("add_to_memory")
            )
        elif isinstance(model, FormatMemoryActionStepModel):
            return FormatMemoryActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                format_memory = kwargs.get("format_memory")
            )
        elif isinstance(model, EvaluateActionStepModel):
            return EvaluateActionStep(
                model, 
                global_context = kwargs.get("global_context"),
                add_to_context = kwargs.get("add_to_context"),
                llm_call = kwargs.get("llm_call"),
                json_llm_call = kwargs.get("json_llm_call"),
            )
        else:
            raise ValueError(f"No logic builder registered for model {type(model).__name__}")