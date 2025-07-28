import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Título principal
st.title("📊 Dashboard de Predicción de Demanda Notarial")
st.markdown("---")

# Cargar datos procesados
@st.cache_data
def cargar_datos():
    df_2024 = pd.read_csv("demanda_predicha_2024.csv") if os.path.exists("demanda_predicha_2024.csv") else None
    df_2025 = pd.read_csv("demanda_predicha_2025.csv") if os.path.exists("demanda_predicha_2025.csv") else None
    return df_2024, df_2025

df_2024, df_2025 = cargar_datos()

year_option = st.selectbox("Selecciona el año para visualizar predicciones:", [2024, 2025])

if year_option == 2024 and df_2024 is not None:
    df = df_2024.copy()
elif year_option == 2025 and df_2025 is not None:
    df = df_2025.copy()
else:
    st.warning("No se encontraron datos para ese año.")
    st.stop()

# Conversión de fecha y agregaciones
if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes"] = df["fecha"].dt.month
    df["semana"] = df["fecha"].dt.isocalendar().week

    mensual = df.groupby("mes")["demanda_predicha"].sum().reset_index()
    semanal = df.groupby("semana")["demanda_predicha"].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demanda mensual")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=mensual, x="mes", y="demanda_predicha", ax=ax1)
        ax1.set_title("Demanda mensual predicha")
        ax1.set_xlabel("Mes")
        ax1.set_ylabel("Demanda")
        st.pyplot(fig1)

    with col2:
        st.subheader("Demanda semanal")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=semanal, x="semana", y="demanda_predicha", marker="o", ax=ax2)
        ax2.set_title("Demanda semanal predicha")
        ax2.set_xlabel("Semana")
        ax2.set_ylabel("Demanda")
        st.pyplot(fig2)

    # Total
    st.success(f"Demanda total estimada en {year_option}: {df['demanda_predicha'].sum():.2f}")

else:
    st.error("El archivo no contiene la columna 'fecha'. Verifica que el archivo de entrada esté correcto.")
