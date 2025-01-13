import streamlit as st
from procesamiento import index, analizar_comparacion, analizar_plazos

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Reglamentos",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Título principal
st.title("📊 **Análisis de Reglamentos y Normativas**")
st.markdown(
    "Esta aplicación realiza análisis comparativos entre el **Reglamento Interno del Fondo** "
    "y la **Normativa Aplicable** para determinar su cumplimiento normativo."
)

# Inicializar historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Contenedor de botones
col1, col2, col3 = st.columns([1, 1, 1])

# Botón para generar respuesta de comparación
with col1:
    if st.button("💡 Generar Respuesta de Comparación"):
        user_question = "¿El reglamento cumple los requisitos para repartir dividendos?"
        st.session_state.chat_history.insert(0, {"role": "user", "content": user_question})
        respuesta = analizar_comparacion(index)
        st.session_state.chat_history.insert(1, {"role": "assistant", "content": respuesta})

# Botón para generar respuesta de plazos
with col2:
    if st.button("🕒 Generar Respuesta de Plazos"):
        user_question = "¿Cuál es el plazo de duración del fondo y el vencimiento de la deuda?"
        st.session_state.chat_history.insert(0, {"role": "user", "content": user_question})
        respuesta = analizar_plazos(index)
        st.session_state.chat_history.insert(1, {"role": "assistant", "content": respuesta})

# Botón para limpiar el chat
with col3:
    if st.button("🧹 Limpiar Chat"):
        st.session_state.chat_history = []

# Contenedor de mensajes
st.header("💬 **Chat de Análisis**")

# Mostrar el historial de chat en bloques con `st.chat_message`
for i in range(0, len(st.session_state.chat_history), 2):
    if i < len(st.session_state.chat_history):
        user_message = st.session_state.chat_history[i]
        with st.chat_message("user"):
            st.write(user_message["content"])
    if i + 1 < len(st.session_state.chat_history):
        assistant_message = st.session_state.chat_history[i + 1]
        with st.chat_message("assistant"):
            st.write(assistant_message["content"])