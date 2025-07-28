# exploracion.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/dataset_servicios_normalizado.csv")

@st.cache_data
def cargar_correlacion():
    return pd.read_csv("data/mapa_correlaciones_post_procesamiento.csv", index_col=0)

st.title("🔍 Exploración de Datos")

df = cargar_dataset()
st.subheader("📄 Vista general del dataset")

# Filtros por columnas si existen
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'])
    fecha_inicio = st.date_input("Fecha desde:", df['fecha'].min())
    fecha_fin = st.date_input("Fecha hasta:", df['fecha'].max())
    df = df[(df['fecha'] >= pd.to_datetime(fecha_inicio)) & (df['fecha'] <= pd.to_datetime(fecha_fin))]

if 'servicio' in df.columns:
    servicio_seleccionado = st.multiselect("Filtrar por servicio:", df['servicio'].unique())
    if servicio_seleccionado:
        df = df[df['servicio'].isin(servicio_seleccionado)]

st.dataframe(df, use_container_width=True)

st.subheader("🔗 Mapa de correlaciones")
corr = cargar_correlacion()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
