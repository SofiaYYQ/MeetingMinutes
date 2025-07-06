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
