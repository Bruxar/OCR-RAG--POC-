# Proyecto OCR con PaddleOCR y Pinecone

Este proyecto realiza OCR (Reconocimiento Óptico de Caracteres) en documentos PDF escaneados utilizando la librería PaddleOCR. Los documentos procesados se dividen en fragmentos (chunking) y se introducen en una base de datos vectorial en Pinecone. Posteriormente, se pueden realizar consultas sobre estos datos para obtener análisis utilizando la API de OpenAI con el modelo GPT-4o-mini.

## Características

- **OCR con PaddleOCR**: Extrae texto de documentos PDF escaneados.
- **Chunking**: Divide los documentos en fragmentos manejables.
- **Base de datos vectorial en Pinecone**: Almacena los fragmentos de texto para consultas eficientes.
- **Análisis con OpenAI GPT-4o-mini**: Realiza consultas y análisis avanzados sobre los datos almacenados.
