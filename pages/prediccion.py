# prediccion.py

import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def cargar_pred_mensual_2024():
    return pd.read_csv("data/demanda_predicha_mensual_2024.csv")

@st.cache_data
def cargar_pred_mensual_2025():
    return pd.read_csv("data/demanda_predicha_mensual_2025.csv")

@st.cache_data
def cargar_pred_semanal_2024():
    return pd.read_csv("data/demanda_predicha_semanal_2024.csv")

@st.cache_data
def cargar_pred_semanal_2025():
    return pd.read_csv("data/demanda_predicha_semanal_2025.csv")

@st.cache_data
def cargar_pred_total_2024():
    return pd.read_csv("data/demanda_predicha_2024.csv")

@st.cache_data
def cargar_pred_total_2025():
    return pd.read_csv("data/demanda_predicha_2025.csv")

st.title("📅 Predicción de Demanda Futura")

anio = st.selectbox("Selecciona el año a visualizar:", [2024, 2025])

# Mensual
st.subheader("📆 Predicción Mensual")
if anio == 2024:
    df_mensual = cargar_pred_mensual_2024()
else:
    df_mensual = cargar_pred_mensual_2025()

fig_mensual = px.bar(df_mensual, x="mes", y="demanda_predicha",
                     labels={"mes": "Mes", "demanda_predicha": "Demanda Predicha"},
                     title=f"Demanda Predicha por Mes - {anio}")
st.plotly_chart(fig_mensual, use_container_width=True)

# Semanal
st.subheader("📈 Predicción Semanal")
if anio == 2024:
    df_semanal = cargar_pred_semanal_2024()
else:
    df_semanal = cargar_pred_semanal_2025()

fig_semanal = px.line(df_semanal, x="semana", y="demanda_predicha",
                      markers=True,
                      labels={"semana": "Semana", "demanda_predicha": "Demanda Predicha"},
                      title=f"Demanda Predicha por Semana - {anio}")
st.plotly_chart(fig_semanal, use_container_width=True)

# Total anual
st.subheader("📊 Total Anual")
if anio == 2024:
    df_total = cargar_pred_total_2024()
else:
    df_total = cargar_pred_total_2025()

st.metric(label=f"Total Predicho para {anio}", value=int(df_total['demanda_predicha'].sum()))
