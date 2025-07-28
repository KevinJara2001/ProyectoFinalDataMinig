# evaluacion.py
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def cargar_importancia():
    return pd.read_csv("data/importancia_variables.csv")

@st.cache_data
def cargar_errores():
    return pd.read_csv("data/errores_por_fold.csv")

st.title("📊 Evaluación del Modelo")

# Importancia de variables
st.subheader("🔎 Importancia de Variables")
df_imp = cargar_importancia()
st.write("🧪 Columnas disponibles:", df_imp.columns.tolist())

# Detectar columnas automáticamente
col_importancia = [col for col in df_imp.columns if "import" in col.lower()]
col_variable = [col for col in df_imp.columns if "var" in col.lower() or "feature" in col.lower()]

if col_importancia and col_variable:
    importancia = df_imp.sort_values(col_importancia[0], ascending=True)
    
    fig_imp = px.bar(importancia, 
                     x=col_importancia[0], 
                     y=col_variable[0], 
                     orientation="h", 
                     color=col_importancia[0], 
                     color_continuous_scale="Blues")
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.error("❌ No se encontraron columnas de importancia y variable en el CSV.")

# Errores por fold
st.subheader("📉 Errores por Fold")
errores = cargar_errores()
errores.columns = errores.columns.str.strip()
errores = errores.rename(columns={
    "Fold": "fold",
    "MAE": "mae",
    "MSE": "mse"
})

st.write("📋 Columnas detectadas en errores_por_fold.csv:", errores.columns.tolist())

# Renombrar para estandarizar
errores = errores.rename(columns={
    "Fold": "fold",
    "MAE": "mae",
    "MSE": "mse"
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=errores["fold"],
    y=errores["mae"],
    mode='lines+markers',
    name='MAE por fold',
    marker=dict(symbol='circle'),
    line=dict(color='royalblue')
))

fig.add_trace(go.Scatter(
    x=errores["fold"],
    y=errores["mse"],
    mode='lines+markers',
    name='MSE por fold',
    marker=dict(symbol='square'),
    line=dict(color='darkorange')
))

fig.update_layout(
    title="Errores por Fold - Validación Cruzada",
    xaxis_title="Fold",
    yaxis_title="Error",
    legend_title="Métrica",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)