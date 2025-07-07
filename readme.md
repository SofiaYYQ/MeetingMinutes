questions
    prompts: prompts para mejorar las respuestas de salida para facilitar la comparación posterior
    reports:

## Ejecución con Streamlit
```bash
streamlit run main_chatbot.py
```

Spacy
python -m spacy download es_core_news_sm

## Ejecución de pruebas con `pytest`

Desde el directorio raíz del proyecto, ejecuta el siguiente comando en la terminal:

```bash
pytest
```
Se necesitan pytest y pytest-cov.

## Fuente de Stopwords en Español

La lista de *stopwords* utilizada en este archivo proviene del repositorio oficial de stopwords-iso, un proyecto colaborativo que recopila listas de palabras vacías (*stopwords*) para múltiples idiomas.

Repositorio: https://github.com/stopwords-iso/stopwords-es

Este recurso es útil para tareas de procesamiento de lenguaje natural (NLP), como filtrado de palabras comunes que no aportan significado relevante en análisis de texto. En este caso, se utiliza en la extracción de palabras clave.

## uso del dsl
query documents estan ya en el contexto global, asi que se pueden usar en prompts {}, 

los steps, el resultado se guardan en output. si es composite, se guarda el ultimo output de los steps.

si no hay resultado step por ejemplo en if false que es opcional.devuelve none y la clase princpal se encarga de devolver la anterior si existe. sino devuelve "no output"

apply filters devuelve un objeto cuyo primer indice son los filtersd docs y el segundo indeice unmatch values.

step_type: set variable
source: valor que va a haber en la variable, expresion python. puede sacar del contexto,


una posible implementacion a partir de observaciones de patrones en el caso de uso
el tema de seguridad no esta contemplada

solo acepta salidas json de un nivel de anidamiento y solo objeto no lista. los metadatos estan excluidos