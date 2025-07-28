import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Comparación de Modelos", layout="centered")
st.title("📊 Comparación Interactiva entre Modelos")

# Cargar CSV
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/comparacion_modelos.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = cargar_datos()

# Renombrar columnas
renombrar = {}
for col in df.columns:
    if "modelo" in col:
        renombrar[col] = "Modelo"
    elif "mae" in col:
        renombrar[col] = "MAE"
    elif "rmse" in col:
        renombrar[col] = "RMSE"
    elif "mse" in col:
        renombrar[col] = "MSE"
    elif col in ["r2", "r²", "r_2"]:
        renombrar[col] = "R2"

df = df.rename(columns=renombrar)

# Columnas métricas disponibles
metricas = [col for col in ["MAE", "RMSE", "MSE", "R2"] if col in df.columns]

# Selección de métrica
metrica = st.selectbox("📈 Selecciona la métrica:", metricas)

# Diccionario para títulos bonitos
titulos = {
    "MAE": "MAE - Error Absoluto Medio",
    "RMSE": "RMSE - Raíz del Error Cuadrático Medio",
    "MSE": "MSE - Error Cuadrático Medio",
    "R2": "R² - Coeficiente de Determinación"
}

# Gráfico interactivo con Plotly
if metrica:
    fig = px.bar(
        df,
        x="Modelo",
        y=metrica,
        title=titulos[metrica],
        text_auto=".2f",
        labels={metrica: titulos[metrica], "Modelo": "Modelo"},
        color="Modelo"
    )
    fig.update_layout(xaxis_title="Modelo", yaxis_title=metrica)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No se encontró ninguna métrica para graficar.")
