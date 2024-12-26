from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv

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

def query_index(index, query, top_k=3):
    # Realiza una consulta en Pinecone y devuelve el contexto combinado
    query_embedding = get_embeddings(query)
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    context = [
        x['metadata']['text'] for x in results['matches']
    ]
    return context

def ask_openai(query, context):
    # Consulta al modelo de OpenAI con el contexto combinado
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Actúa como un experto en normativas legales."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta:\n{query}"}
        ]
    )
    return completion.choices[0].message.content

# Uso del programa
if __name__ == "__main__":
    # Paso 1: Extraer texto del PDF
    input_pdf = "PDF A PROCESAR.pdf"
    texto_extraido = extract_text_from_pdf(input_pdf)

    #Paso 2: Indexar texto en Pinecone
    document_id = "ID DEL DOCUMENTO"
    index_documents(index, document_id, texto_extraido)

    # Paso 3: Realizar consulta
    pregunta = "ACA VA TU PREGUNTA :)"
    print("Realizando consulta en Pinecone...")
    contexto = query_index(index, pregunta)
    print(f"Contexto recuperado:\n{contexto}")

    # Paso 4: Consultar OpenAI
    print("Consultando OpenAI...")
    respuesta = ask_openai(pregunta, contexto)
    print(f"Respuesta:\n{respuesta}")

