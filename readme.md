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

## 🛠 Instalación necesaria

Para poner en marcha el proyecto, asegúrate de seguir estos pasos:

### 1. Instalar Ollama localmente
- Permite ejecutar modelos de lenguaje en tu máquina sin necesidad de conexión externa.
- Consulta la documentación oficial para tu sistema operativo: https://ollama.com

### 2. Instalar Python
- Asegúrate de tener Python instalado (recomendado: versión 3.10 o superior).
- Puedes verificarlo con:
  ```bash
  python --version
  ```

### 3. Instalar dependencias del proyecto
- Desde la raíz del proyecto, ejecuta en la terminal:
  ```bash
  pip install -r requirements.txt
  ```

## Ejecución del procedimiento de evaluación
Existe 3 configuraciones distintas para evaluar los modelos. Antes de la ejecución del comando, es necesario establecer el modelo a evaluar dentro de cada fichero correspondiente. 

1. Usa la configuración por defecto
```bash
python src\main_base_model_evaluation.py
```

2. Usa el agente de Llamaindex
```bash
python src\main_reactagent_evaluation.py
```

3. Usa el enfoque propuesto
- Con DSL
```bash
python src\main_workflow.py
```
- Sin DSL
```bash
python src\main_workflow_without_dsl.py
```

## Ejecución del agente con interfaz streamlit
  ```bash
  streamlit run src\main_chatbot_without_dsl.py
  ```

## Ejecución de pruebas con `pytest`

Desde el directorio raíz del proyecto, ejecuta el siguiente comando en la terminal:

```bash
pytest
```
Se necesitan tener instalados los paquetes pytest y pytest-cov.