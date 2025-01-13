import streamlit as st
from procesamiento import index, extract_text_from_pdf, get_all_document_ids, index_documents, analizar_comparacion, analizar_plazos
import os
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Fondos",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar para subir y procesar PDFs
st.sidebar.header("📂 Subir y Procesar PDF")

# Subir archivo
uploaded_file = st.sidebar.file_uploader("Sube un nuevo archivo PDF", type=["pdf"])

if uploaded_file:
    st.sidebar.write(f"📄 Archivo subido: **{uploaded_file.name}**")

    # Boton para procesar el archivo
    if st.sidebar.button("Procesar e Indexar"):
        with st.spinner("📖 Extrayendo texto con OCR..."):
            # Guardar el archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name

            # Extraer texto mediante OCR
            extracted_text = extract_text_from_pdf(temp_pdf_path)

        with st.spinner("🔄 Indexando en Pinecone..."):
            # Generar document_id dinámico basado en el nombre del archivo
            document_id = os.path.splitext(uploaded_file.name)[0]

            index_documents(index, document_id, extracted_text)

        # Mensaje de éxito
        st.sidebar.success(f"✅ Documento indexado exitosamente con ID: **{document_id}**")

# Recuperar document_id desde Pinecone
with st.sidebar:
    st.header("⚙️ Configuración de Análisis")
    st.write("Selecciona el fondo y la normativa aplicable para realizar los análisis.")

    # Recuperar todos los document_id dinamicamente
    document_ids = get_all_document_ids(index)

    # Dropdowns dinamicos
    fondo = st.selectbox("Selecciona el Reglamento del Fondo", options=document_ids, key="fondo")
    normativa = st.selectbox("Selecciona la Normativa Aplicable", options=document_ids, key="normativa")

    # Mensaje de confirmación
    st.write(f"📄 Fondo seleccionado: **{fondo}**")
    st.write(f"⚖️ Normativa seleccionada: **{normativa}**")

    # Boton para actualizar
    if st.button("Actualizar Selección"):
        st.session_state.selected_fondo = fondo
        st.session_state.selected_normativa = normativa

# Mostrar la selección actual
st.markdown("### 📄 Información de Análisis Seleccionada")
st.write(f"**Fondo Actual:** {st.session_state.get('selected_fondo', 'No seleccionado')}")
st.write(f"**Normativa Actual:** {st.session_state.get('selected_normativa', 'No seleccionada')}")

# Título principal
st.title("📊 **Análisis de Reglamentos y Normativas**")
st.markdown(
    "Esta aplicación realiza análisis comparativos entre el **Reglamento Interno del Fondo** "
    "y la **Normativa Aplicable** para determinar su cumplimiento normativo."
)

# Inicializar historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_fondo" not in st.session_state:
    st.session_state.current_fondo = None
if "current_normativa" not in st.session_state:
    st.session_state.current_normativa = None

# Contenedor de botones
col1, col2, col3 = st.columns([1, 1, 1])

# Botón para generar respuesta de comparación
with col1:
    if st.button("💡 Generar Respuesta de Comparación"):
        user_question = "¿El reglamento cumple los requisitos para repartir dividendos?"
        respuesta, fondo, normativa = analizar_comparacion(index)
        st.session_state.chat_history.insert(0, {"role": "user", "content": user_question})
        st.session_state.chat_history.insert(1, {"role": "assistant", "content": respuesta})
        st.session_state.current_fondo = fondo
        st.session_state.current_normativa = normativa

# Botón para generar respuesta de plazos
with col2:
    if st.button("🕒 Generar Respuesta de Plazos"):
        user_question = "¿Cuál es el plazo de duración del fondo y el vencimiento de la deuda?"
        respuesta, fondo = analizar_plazos(index)
        st.session_state.chat_history.insert(0, {"role": "user", "content": user_question})
        st.session_state.chat_history.insert(1, {"role": "assistant", "content": respuesta})
        st.session_state.current_fondo = fondo

# Botón para limpiar el chat
with col3:
    if st.button("🧹 Limpiar Chat"):
        st.session_state.chat_history = []
        st.session_state.current_fondo = None
        st.session_state.current_normativa = None

# Mostrar fondo y normativa actuales
st.markdown("### 📄 Información de Análisis")
if st.session_state.current_fondo and st.session_state.current_normativa:
    st.write(f"**Fondo Actual:** {st.session_state.current_fondo}")
    st.write(f"**Normativa Correspondiente:** {st.session_state.current_normativa}")
else:
    st.write("No hay información de fondo y normativa seleccionada.")

# Contenedor de mensajes con scroll
st.header("💬 **Chat de Análisis**")
st.write("Aquí puedes ver el historial de interacciones con el modelo.")

with st.container():
    for i in range(0, len(st.session_state.chat_history), 2):
        if i < len(st.session_state.chat_history):
            user_message = st.session_state.chat_history[i]
            with st.chat_message("user"):
                st.write(user_message["content"])
        if i + 1 < len(st.session_state.chat_history):
            assistant_message = st.session_state.chat_history[i + 1]
            with st.chat_message("assistant"):
                st.write(assistant_message["content"])