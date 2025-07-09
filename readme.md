## Estructura del Directorio del Proyecto

- `/config`  
  Contiene los archivos de configuración necesarios para la ejecución del proyecto.

- `/data`  
  Incluye los documentos en formato PDF que sirven como base de conocimiento a consultar por los modelos.

- `/dataset`  
  Conjunto de datos utilizado para la evaluación:
  - `/answers`: Contiene las respuestas esperadas respecto a las preguntas.
  - `/questions`: Contiene el conjunto de preguntas utilizadas en la evaluación.
    - `/prompts`: Cotiene prompts para extraer respuestas estructuradas a partir de respuestas textuales, facilitando la comparación automatizada posterior

- `/logs`  
  Almacena los registros de ejecución del sistema.

- `/reports`  
  Contiene los informes generados en formatos Excel y CSV, que resumen los resultados de la evaluación.

- `/results`  
  Guarda las respuestas generadas por los modelos tras el proceso de evaluación.

- `/src`  
  Directorio principal del código fuente del proyecto.

- `.env`
  Es necesario establecer `PYTHONPATH=src` en este fichero para que los comandos posteriores se ejecuten correctamente, ya que esto permite que Python reconozca el directorio de código fuente como parte del entorno de búsqueda de módulos.


## Evaluación
Existe 3 configuraciones distintas para evaluar los modelos. Antes de la ejecución del comando, es necesario establecer el modelo a evaluar dentro de cada fichero correspondiente. 

1. Usa la configuración por defecto
```bash
python main_base_model_evaluation.py
```

2. Usa el agente de Llamaindex
```bash
python main_reactagent_evaluation.py
```

3. Usa el enfoque propuesto
- Con DSL
```bash
python main_workflow.py
```
- Sin DSL
```bash
python main_workflow_without_dsl.py
```

## Ejecución de pruebas con `pytest`

Desde el directorio raíz del proyecto, ejecuta el siguiente comando en la terminal:

```bash
pytest
```
Se necesitan tener instalados los paquetes pytest y pytest-cov.

## Notas sobre el comportamiento del workflow construido por el DSL (Desactualizado)

### Contexto Global
- Los documentos de consulta (`query documents`) ya están disponibles en el contexto global.
- Pueden utilizarse directamente en los prompts mediante la sintaxis `{}`.

### Ejecución de Pasos (`steps`)
- Cada paso genera un resultado que se almacena en `output`.
- En pasos compuestos (`composite`), solo se guarda la salida del último paso.
- Si un paso no produce resultado (por ejemplo, en una condición `if` que no se cumple), se devuelve `None`.
- La clase principal se encarga de devolver el resultado anterior si existe; en caso contrario, retorna `"no output"`.

### Aplicación de Filtros (`apply filters`)
- Devuelve una tupla con dos elementos:
  1. Documentos que cumplen con los filtros (`filtered docs`).
  2. Valores que no coinciden (`unmatched values`).

### Tipo de Paso: `set variable`
- El campo `source` define el valor que se asignará a la variable.
- Puede ser una expresión en Python y puede extraerse del contexto.

### Limitaciones y Consideraciones
- Esta implementación se basa en patrones observados en el caso de uso.
- **No se han considerado aspectos de seguridad.**
- Solo se aceptan salidas en formato JSON:
  - Un único nivel de anidamiento.
  - Solo objetos (no listas).
- Los metadatos están excluidos del procesamiento.

### Memoria (`add_to_memory`)
- Solo el campo `description` se formatea.
- Puede incluir variables mediante la sintaxis `{variable}`.

### Evaluación (`evaluation`)
- En el DSL actual, si la evaluación siempre determina que una respuesta es "inventada", puede producir un bucle infinito.
- Este comportamiento debe ser controlado para evitar errores en la ejecución.
