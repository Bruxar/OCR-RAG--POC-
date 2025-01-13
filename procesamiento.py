from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

# Claves de API
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Conectar al indice de Pinecone
index = pc.Index(INDEX_NAME)

def extract_text_selectable(input_pdf):
    # Extrae texto directamente de un PDF con texto seleccionable
    reader = PdfReader(input_pdf)
    extracted_text = []
    for page in reader.pages:
        extracted_text.append(page.extract_text().strip())
    return "\n".join(extracted_text)

# Funciones de OCR y texto
def extract_text_from_pdf(input_pdf):
    # Extrae texto desde un PDF escaneado utilizando PaddleOCR
    ocr = PaddleOCR(lang='es')
    images = convert_from_path(input_pdf, dpi=300)
    extracted_text = []

    for i, image in enumerate(images):
        print(f"procesando pagina {i+1}...")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(image_cv)
        text = extract_text_from_paddleocr(result)
        extracted_text.append(text)

    return "\n".join(extracted_text)

def extract_text_from_paddleocr(ocr_result):
    # Extrae texto de la salida de PaddleOCR
    extracted_text = []
    for line in ocr_result[0]:
        text, confidence = line[1][0], line[1][1]
        if confidence >= 0.8:
            extracted_text.append(text)
    return "\n".join(extracted_text)

#Funciones Pinecone y OpenAI
def create_chunks(text, chunk_size=200, overlap=50):
    # Divide el texto en chunks con superposicion
    words = text.split()
    chunks = []
    for i in range (0, len(words), chunk_size-overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_embeddings(text):
    # Genera embeddings para el texto utilizando OpenAI
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def index_documents(index, document_id, text):
    # Indexa un documento en Pinecone
    chunks = create_chunks(text)
    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)
        metadata = {
            "document_id": document_id,
            "chunk_id": i,
            "text": chunk
        }
        index.upsert([(f"{document_id}_{i}", embedding, metadata)])

def query_index(index, query, tag, top_k=3):
    # Realiza una consulta en Pinecone y devuelve el contexto combinado
    query_embedding = get_embeddings(query)
    results = index.query(
        query_embedding,
        filter={
            "document_id": {"$eq": tag}
        },
        top_k=top_k,
        include_metadata=True)
    context = [
        x['metadata']['text'] for x in results['matches']
    ]
    return context

def get_all_document_ids(index):
    """
    Recupera todos los document_id únicos desde el índice de Pinecone.
    """
    document_ids = set()

    # Define un rango grande basado en el número total de vectores
    response = index.describe_index_stats()
    total_vectors = response.get('total_vector_count', 0)

    # Iterar sobre IDs si son secuenciales o tienen un patrón predecible
    for i in range(total_vectors):
        vector_id = str(i)  # Si tus IDs no son secuenciales, ajusta este patrón
        try:
            response = index.fetch(ids=[vector_id])
            vectors = response.get('vectors', {})

            for vector_data in vectors.values():
                metadata = vector_data.get('metadata', {})
                if 'document_id' in metadata:
                    document_ids.add(metadata['document_id'])

        except Exception as e:
            print(f"Error al obtener el vector {vector_id}: {e}")

    return list(document_ids)

def ask_openai(query):
    # Consulta al modelo de OpenAI con el contexto combinado
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Actúa como un experto en normativas legales."},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

def obtener_contexto(index, etiqueta, consulta, top_k=4):
    # Recupera contexto desde Pinecone filtrando por metadata.
    print(f"Realizando consulta en Pinecone para la etiqueta: {etiqueta}...")
    resultados = query_index(index, consulta, etiqueta, top_k)
    return "\n".join(resultados)

def construir_prompt(contexto_reglamento, contexto_normativa):
    # Construye el prompt para enviar al modelo GPT.
    prompt = f"""
    Contexto 1: Reglamento Interno del Fondo
    {contexto_reglamento}

    Contexto 2: Normativa Aplicable al Fondo
    {contexto_normativa}

    Tarea: Realiza un análisis comparativo entre el Reglamento Interno del Fondo y la Normativa Aplicable para determinar si el Fondo cumple con los requisitos necesarios para realizar el reparto de dividendos definitivos y provisorios, debes responder si cumple o no. En tu análisis, identifica específicamente:
    1. Las condiciones establecidas en el Reglamento Interno sobre el reparto de dividendos.
    2. Los requisitos impuestos por la Normativa Aplicable para permitir dicho reparto.
    3. Si existen discrepancias entre ambos documentos, explícitalas.
    """
    return prompt

def construir_prompt_plazos(contexto_duracion, contexto_vencimiento):
    # Construye el prompt para enviar al modelo GPT.
    prompt = f"""
    Contexto 1: Plazo de Duración del Fondo
    {contexto_duracion}

    Contexto 2: Vencimiento en el Pago de la Deuda con CORFO
    {contexto_vencimiento}

    Tarea: Realiza un análisis de los contextos para obtener el plazo de duracion del fondo y el vencimiento en el pago de la deuda con CORFO, tu respuesta debe ser breve. En tu análisis, identifica específicamente:
    1. El plazo de duración del fondo establecido en el Reglamento Interno.
    2. La fecha de vencimiento en el pago de la deuda con CORFO.
    3. Si existen discrepancias entre ambos plazos, explícitalas.
    """
    return prompt

def analizar_comparacion(index, top_k=4):
    #Realiza comparacion entre reglamento interno y normativa aplicable
    
    fondo = "fondo_devlabs_20210504"
    normativa = "normFET_138"

    # Paso 1: Obtener contexto para reglamento interno y normativa aplicable
    busqueda = "reparto de dividendos definitivos y provisorios"
    contexto_reglamento = obtener_contexto(index, fondo, busqueda, top_k)
    contexto_normativa = obtener_contexto(index, normativa, busqueda, top_k)

    # Paso 2: Construir prompt
    prompt = construir_prompt(contexto_reglamento, contexto_normativa)
    print(prompt)

    # Paso 3: Realizar consulta a OpenAI
    respuesta = ask_openai(prompt)

    return respuesta, fondo, normativa

def analizar_plazos(index, top_k=4):
    #Realiza analisis de plazo de duracion del fondo y vencimiento en el pago de la deuda con CORFO
    busqueda_duracion = "plazo de duración del fondo"
    busqueda_vencimiento = "vencimiento en el pago de la deuda con CORFO"

    fondo = "fondo_devlabs_20210504"

    contexto_duracion = obtener_contexto(index, fondo, busqueda_duracion, top_k)
    contexto_vencimiento = obtener_contexto(index, fondo, busqueda_vencimiento, top_k)

    prompt = construir_prompt_plazos(contexto_duracion, contexto_vencimiento)
    print(prompt)

    respuesta = ask_openai(prompt)

    return respuesta, fondo

__all__ = [
    "extract_text_from_pdf",
    "index_documents",
    "query_index",
    "analizar_comparacion",
    "analizar_plazos"
]