import json

class DSLExecutor:
    def __init__(self, dsl):
        self.dsl = dsl
        self.context = {}

    def execute(self):
        for step in self.dsl['pipeline']:
            self.execute_step(step)

    def execute_step(self, step):
        if 'if' in step:
            condition = step['if']
            if not self.evaluate_condition(condition):
                return

        action = step.get('action')
        
        if 'foreach' in step and step['foreach']:
            source = step['foreach']
            iterable = self.context.get(source) or self.dsl['inputs'].get(source)
            
            if iterable is None:
                raise ValueError(f"No se encontró la variable '{source}' en el contexto ni en los inputs.")
            
            if action:
                if action == 'extract_text':
                    self.context[step['output']] = ["Texto del documento 1", "Texto del documento 2", "Texto del documento 3"]
                elif action == 'filter':
                    self.context[step['output']] = self.filter(step)
                elif action == 'choose':
                    self.choose(step)
                elif action == 'split_by_label':
                    self.split_by_label(step)
                elif action == 'if_then_else':
                    self.if_then_else(step)
            else:
                results = []
                
                if step['id'] == "evaluate_quality":
                    self.context[step['output']] = [8, 7, 3]
                elif step['id'] == "evaluate_originality":
                    self.context[step['output']] = [7, 7, 7]
                elif step['id'] == "evaluate_relevance":
                    self.context[step['output']] = [6, 7, 9]
                elif step['id'] == "verify_factuality":
                    self.context[step['output']] = ["explícita", "inventada"]
                elif step['id'] == "classify_affirmation":
                    self.context[step['output']] = ["afirmativa"]
                else:
                    for i, item in enumerate(iterable):
                        if 'with' in step:
                            with_item = self.context[step["with"]]
                            result = self.prompt(step, item, with_item[i])
                        else:
                            result = self.prompt(step, item)
                        results.append(result)
                    self.context[step['output']] = results
                
        else:
            if action:
                if action == 'extract_text':
                    self.context[step['output']] = ["Texto del documento 1", "Texto del documento 2", "Texto del documento 3"]
                elif action == 'filter':
                    self.context[step['output']] = self.filter(step)
                elif action == 'choose':
                    self.choose(step)
                elif action == 'split_by_label':
                    self.split_by_label(step)
                elif action == 'if_then_else':
                    self.if_then_else(step)
            else:
                if step['id'] == "classify_query":
                    self.context[step['output']] = "agregada"
                elif step['id'] == "generate_subquery":
                    self.context[step['output']] = "query transformado"
                else:
                    self.context[step['output']] = self.prompt(step)
                

    def evaluate_condition(self, condition):
        return eval(condition, {}, self.context)

    def extract_text(self, step):
        # Dummy implementation for extracting text from documents
        self.context[step['output']] = ["Texto del documento 1", "Texto del documento 2", "Texto del documento 3"]

    def filter(self, step):
        inputs = [self.context[input] for input in step['inputs']]
        condition = step['condition']
        
        # filtered = [item for item in zip(*inputs) if eval(condition, {}, dict(zip(step['inputs'], item)))]
        
        
        filtered = []
        zipped_inputs = zip(*inputs)

        for item in zipped_inputs:
            # Crear un diccionario que asocia cada nombre de entrada con su valor correspondiente
            local_vars = dict(zip(step['inputs'], item))
            
            # Evaluar la condición usando eval con el diccionario como contexto local
            if eval(condition, {}, local_vars):
                filtered.append(item)


        return [item[-1] for item in filtered]

    def choose(self, step):
        condition = step['condition']
        self.context[step['output']] = self.context[step['if_true']] if eval(condition, {}, self.context) else self.context[step['if_false']]

    def split_by_label(self, step):
        items = self.context[step['inputs'][0]]
        labels = self.context[step['inputs'][1]]
        result = {label: [] for label in step['labels']}
        for item, label in zip(items, labels):
            if label in result:
                result[label].append(item)
        for label, output in step['outputs'].items():
            self.context[output] = result[label]

    def if_then_else(self, step):
        condition = step['condition']
        if eval(condition, {}, self.context):
            for substep in step['if_true']:
                self.execute_step(substep)
        else:
            for substep in step['if_false']:
                self.execute_step(substep)

    def prompt(self, step, item = None, with_item = None):
        # Dummy implementation for calling LLM
        prompt = step['prompt']
        if item:
            prompt = prompt.replace(f"{{{{{'item'}}}}}", str(item))
        
        if with_item:
            prompt = prompt.replace(f"{{{{{'with_item'}}}}}", str(with_item))
        for key, value in self.context.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        # self.context[step['output']] = f"Respuesta generada para el prompt: {prompt}"
        return f"Respuesta generada para el prompt: {prompt}"

    # def prompt_list(self, step):
    #     # Dummy implementation for calling LLM
    #     prompt = step['prompt']
    #     if step['output'] == "filtered_texts":
    #         return 8

        
        

# Ejemplo de DSL
dsl = {
    "inputs": {
        "query": "¿Cuál fue el total de ventas por categoría en 2024?",
        "documents": [
"""
Documento ACTA 1.pdf:
fecha(Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA.): 24/02/2025
num_asistentes(Es el número de asistentes a la reunión): 20
lista_asistentes(Es la lista de nombres de los asistentes de la reunión. ):
- Juan Pérez Gutiérrez
- Marta González Ramírez
- Luis Ramírez Ortega
- Ana Sánchez Herrera
- Roberto Martínez Vázquez
- Carmen Herrera Jiménez
- Pedro Jiménez Suárez
- Laura Díaz Castro
- Manuel Ortega Medina
- Isabel Castro Torres
- Jorge Moreno Navarro
- Beatriz Suárez Aguilar
- Alejandro Torres Rojas
- Natalia Vázquez Gutiérrez
- Eduardo Rojas Martínez
- Silvia Medina Pérez
- Ricardo Flores Sánchez
- Patricia Navarro Díaz
- Daniel Gutiérrez Moreno
- Rosa Aguilar Fernández
secretario(Es el nombres del secretorio o secretaria de la reunión. ): Rosa Aguilar Fernández
presidente(Es el nombres del presidente o presidenta de la reunión.): Juan Pérez Gutiérrez
""",
"""
Documento ACTA 2.pdf:
fecha(Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA.): 25/02/2025
num_asistentes(Es el número de asistentes a la reunión): 20
lista_asistentes(Es la lista de nombres de los asistentes de la reunión. ):
- Antonio Martínez López
- Carmen Herrera Jiménez
- Luis Ramírez Ortega
- Isabel Castro Torres
- Jorge Moreno Navarro
- Beatriz Suárez Aguilar
- Eduardo Rojas Martínez
- Silvia Medina Pérez
- Ricardo Flores Sánchez
- Patricia Navarro Díaz
- Daniel Gutiérrez Moreno
- Rosa Aguilar Fernández
- Francisco Torres Delgado
- Laura Díaz Castro
- Marta González Ramírez
- Juan Pérez Gutiérrez
- Ana Sánchez Herrera
- Pedro Jiménez Suárez
- Roberto Martínez Vázquez
- Natalia Vázquez Gutiérrez
secretario(Es el nombres del secretorio o secretaria de la reunión. ): Natalia Vázquez Gutiérrez
presidente(Es el nombres del presidente o presidenta de la reunión.): Antonio Martínez López
""",
"""
Documento ACTA 3.pdf:
 fecha(Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA.): 25/08/2025
num_asistentes(Es el número de asistentes a la reunión): 18
lista_asistentes(Es la lista de nombres de los asistentes de la reunión. ):
- Beatriz Suárez Aguilar
- Manuel Ortega Medina
- Isabel Castro Torres
- Jorge Moreno Navarro
- Patricia Navarro Díaz
- Eduardo Rojas Martínez
- Silvia Medina Pérez
- Ricardo Flores Sánchez
- Daniel Gutiérrez Moreno
- Rosa Aguilar Fernández
- Laura Díaz Castro
- Marta González Ramírez
- Antonio Martínez López
- Alejandro Torres Rojas
- Natalia Vázquez Gutiérrez
- Francisco Torres Delgado
- Pedro Jiménez Suárez
- Ana Sánchez Herrera
secretario(Es el nombres del secretorio o secretaria de la reunión. ): Natalia Vázquez Gutiérrez
presidente(Es el nombres del presidente o presidenta de la reunión.): Beatriz Suárez Aguilar
""",
"""
Documento ACTA 5.pdf:
fecha(Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA.): 25/02/2026
num_asistentes(Es el número de asistentes a la reunión): 17
lista_asistentes(Es la lista de nombres de los asistentes de la reunión. ):
- Jorge Moreno Navarro
- Laura Díaz Castro
- Manuel Ortega Medina
- Rosa Aguilar Fernández
- Ricardo Flores Sánchez
- Beatriz Suárez Aguilar
- Pedro Jiménez Suárez
- Ana Sánchez Herrera
- Patricia Navarro Díaz
- Eduardo Rojas Martínez
- Silvia Medina Pérez
- Francisco Torres Delgado
- Daniel Gutiérrez Moreno
- Natalia Vázquez Gutiérrez
- Antonio Martínez López
- Isabel Castro Torres
- Marta González Ramírez
secretario(Es el nombres del secretorio o secretaria de la reunión. ): Natalia Vázquez Gutiérrez
presidente(Es el nombres del presidente o presidenta de la reunión.): Jorge Moreno Navarro
""",
"""
Documento ACTA 6.pdf:
fecha(Es la fecha en la que se celebró la reunión. Está en formato DD/MM/AAAA.): 25/08/2026
num_asistentes(Es el número de asistentes a la reunión): 19
lista_asistentes(Es la lista de nombres de los asistentes de la reunión. ):
- Manuel Ortega Medina
- Beatriz Suárez Aguilar
- Daniel Gutiérrez Moreno
- Rosa Aguilar Fernández
- Ricardo Flores Sánchez
- Pedro Jiménez Suárez
- Ana Sánchez Herrera
- Patricia Navarro Díaz
- Eduardo Rojas Martínez
- Silvia Medina Pérez
- Francisco Torres Delgado
- Natalia Vázquez Gutiérrez
- Antonio Martínez López
- Isabel Castro Torres
- Marta González Ramírez
- Alejandro Torres Rojas
- Jorge Moreno Navarro
- Laura Díaz Castro
- Carmen Herrera Jiménez
secretario(Es el nombres del secretorio o secretaria de la reunión. ): Natalia Vázquez Gutiérrez
presidente(Es el nombres del presidente o presidenta de la reunión.): Manuel Ortega Medina
""",
        ]
    },
    "pipeline": [
        {"id": "extract_text", 
         "foreach": "documents", 
         "action": "extract_text", 
         "output": "texts"
        },
        {
            "id": "evaluate_quality",
            "foreach": "texts",
            "prompt": "Evalúa la calidad del siguiente texto en una escala del 1 al 10:\n---\n{{item}}",
            "model": "gpt-4",
            "output": "quality_scores"
        },
        {
            "id": "evaluate_relevance",
            "foreach": "texts",
            "prompt": "¿Qué tan relevante es este texto para la pregunta \"{{query}}\"?\nResponde con un número del 1 al 10.\n---\n{{item}}",
            "model": "gpt-4",
            "output": "relevance_scores"
        },
        {
            "id": "evaluate_originality",
            "foreach": "texts",
            "prompt": "¿Este texto aporta ideas originales o simplemente repite información común?\nResponde con un número del 1 al 10.\n---\n{{item}}",
            "model": "gpt-4",
            "output": "originality_scores"
        },
        {
            "id": "filter_documents",
            "action": "filter",
            "inputs": [
                "quality_scores",
                "relevance_scores",
                "originality_scores",
                "texts"
            ],
            "condition": "quality_scores >= 7 and relevance_scores >= 5 and originality_scores >= 6",
            "output": "filtered_texts"
        },

        # {"id": "evaluate_quality", 
        #  "foreach": "texts", 
        #  "prompt": "Evalúa la calidad del texto del 1 al 10:\n---\n{{item}}", 
        #  "model": "gpt-4", 
        #  "output": "quality_scores"
        # },
        # {"id": "filter_documents", 
        #  "action": "filter", 
        #  "inputs": ["quality_scores", "texts"], 
        #  "condition": "quality_scores >= 7", 
        #  "output": "filtered_texts"
        # },
        {"id": "classify_query", 
         "prompt": "Clasifica la siguiente consulta como 'agregada' o 'no agregada':\n---\n{{query}}", 
         "model": "gpt-4", 
         "output": "query_type"
        },
        {"id": "generate_subquery", 
         "if": "query_type == 'agregada'", 
         "prompt": "Reformula esta consulta para que sea una subconsulta más precisa:\n---\n{{query}}", 
         "model": "gpt-4", 
         "output": "subquery"
        },
        {"id": "select_query", 
         "action": "choose", 
         "condition": "query_type == 'agregada'", 
         "if_true": "subquery", 
         "if_false": "query", 
         "output": "effective_query"
        },
        {"id": "execute_per_document", 
         "foreach": "filtered_texts", 
         "prompt": "Dado el siguiente documento:\n---\n{{item}}\n---\nResponde a la consulta:\n{{effective_query}}", 
         "model": "gpt-4", 
         "output": "document_answers"
        },
        {"id": "verify_factuality", 
         "foreach": "document_answers", 
         "with": "filtered_texts", # TODO: Revisar
         "prompt": "Evalúa si la siguiente respuesta está basada explícitamente en el documento o si es una inferencia:\n---\nDocumento:\n{{with_item}}\n---\nRespuesta:\n{{item}}\n---\nResponde solo con 'explícita' o 'inferida'.", 
         "model": "gpt-4", 
         "output": "factuality_labels"
        },
        {"id": "filter_explicit", 
         "action": "filter", 
         "inputs": ["factuality_labels", "document_answers"], 
         "condition": "factuality_labels != 'inventada'", 
         "output": "verified_answers"
        },
        {"id": "classify_affirmation", 
         "foreach": "verified_answers", 
         "prompt": "Clasifica la siguiente respuesta como 'afirmativa' o 'negativa':\n---\n{{item}}", 
         "model": "gpt-4", 
         "output": "affirmation_labels"
        },
        {"id": "separate_by_affirmation", 
         "action": "split_by_label", 
         "inputs": ["verified_answers", "affirmation_labels"], 
         "labels": ["afirmativa", "negativa"], 
         "outputs": {"afirmativa": "affirmative_answers", "negativa": "negative_answers"}
        },
        {"id": "decide_final_response", 
         "action": "if_then_else", 
         "condition": "len(affirmative_answers) >= 1", 
         "if_true": [
            {"id": "synthesize_affirmatives", 
             "prompt": "A partir de las siguientes respuestas afirmativas, redacta una respuesta final coherente:\n---\n{{affirmative_answers}}", 
             "model": "gpt-4", 
             "output": "final_answer"
            }
        ], "if_false": [
            {"id": "justify_no_answer", 
             "prompt": "No se puede dar una respuesta afirmativa clara. Justifica esta conclusión usando las siguientes respuestas negativas:\n---\n{{negative_answers}}", 
             "model": "gpt-4", 
             "output": "final_answer"
            }
        ]}
    ]
}

executor = DSLExecutor(dsl)
executor.execute()

print(json.dumps(executor.context, indent=2, ensure_ascii=False))

