# # Diccionario global para registrar constructores
# from config_loader.models import BaseStepModel, CompositeStepModel, FormatDocumentsActionStepModel, LLMCallStepModel
# # from config_loader.steps import FormatDocumentsActionStep, LLMCallStep, Step
# from config_loader.steps import Step
# # class StepLogicFactory:
# #     def __init__(self):
# #         self._registry = {}

# #     def register(self, model_cls, builder_fn):
# #         self._registry[model_cls] = builder_fn

# #     def create(self, model, **kwargs)->Step:
# #         model_cls = model.__class__
# #         builder = self._registry.get(model_cls)
# #         if not builder:
# #             raise ValueError(f"No logic builder registered for model {model_cls.__name__}")
# #         return builder(model, **kwargs)


# # step_logic_factory = StepLogicFactory()


# # step_logic_factory.register(
# #     FormatDocumentsActionStepModel,
# #     lambda model, **kwargs: FormatDocumentsActionStep(model, **kwargs)
# # )

# # step_logic_factory.register(
# #     LLMCallStepModel,
# #     lambda model, **kwargs: LLMCallStep(model, **kwargs)
# # )

# class StepFactory:
#     @staticmethod
#     def create(model: BaseStepModel, **kwargs) -> Step:
#         if isinstance(model, LLMCallStepModel):
#             from config_loader.steps import LLMCallStep
#             return LLMCallStep(
#                 model, 
#                 global_context = kwargs.get("global_context"),
#                 llm_call = kwargs.get("llm_call"),
#                 json_llm_call = kwargs.get("json_llm_call"),
#             )
#         elif isinstance(model, FormatDocumentsActionStepModel):
#             from config_loader.steps import FormatDocumentsActionStep
#             return FormatDocumentsActionStep(
#                 model, 
#                 global_context = kwargs.get("global_context")
#             )
#         elif isinstance(model, CompositeStepModel):
#             from config_loader.steps import CompositeStep
#             return CompositeStep(
#                 model, 
#                 global_context = kwargs.get("global_context"),
#                 llm_call = kwargs.get("llm_call"),
#                 json_llm_call = kwargs.get("json_llm_call"),
#             )
#         else:
#             raise ValueError(f"No logic builder registered for model {type(model).__name__}")