import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MESES_ES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
DIAS_ES = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_servicios_generado_demanda.csv")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["dia_nombre"] = df["dia_semana"].apply(lambda x: DIAS_ES[x])
    df["mes_nombre"] = df["mes"].apply(lambda x: MESES_ES[x-1])
    return df

df = cargar_datos()

st.header("Exploración de Datos - Parte 1")
tabs = st.tabs(["Demanda diaria", "Por día de semana", "Por mes", "Demanda anual"])

with tabs[0]:
    st.markdown("### Distribución diaria de demanda")
    fig, ax = plt.subplots(figsize=(10, 3), dpi=72)
    sns.lineplot(data=df, x="fecha", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

with tabs[1]:
    st.markdown("### Demanda por día de la semana")
    resumen = df.groupby("dia_nombre")["demanda_diaria"].mean().reindex(DIAS_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
    sns.barplot(data=resumen, x="dia_nombre", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.markdown("### Demanda mensual promedio")
    resumen_m = df.groupby("mes_nombre")["demanda_diaria"].mean().reindex(MESES_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
    sns.barplot(data=resumen_m, x="mes_nombre", y="demanda_diaria", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

with tabs[3]:
    st.markdown("### Demanda promedio anual")
    anual = df.groupby("anio")["demanda_diaria"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
    sns.barplot(data=anual, x="anio", y="demanda_diaria", ax=ax)
    st.pyplot(fig)