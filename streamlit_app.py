# streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="Dashboard de Demanda",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Predicción de Demanda")
st.markdown("""
Bienvenido al sistema de análisis y visualización de demanda.
Utiliza el menú de la izquierda para explorar las distintas secciones:

- 🔍 Exploración de datos
- 📊 Evaluación del modelo
- 📈 Comparación real vs predicho
- 📅 Predicción futura
""")
