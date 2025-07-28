# comparacion.py

import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def cargar_diario():
    df = pd.read_csv("data/demanda_real_predicha_por_dia.csv")

    # Mostrar columnas originales
    st.write("📋 Columnas originales:", df.columns.tolist())

    # Forzar renombramiento
    df.columns = ["dia_semana_num", "demanda_real", "demanda_predicha", "dia"]

    return df

@st.cache_data
def cargar_mensual_real():
    return pd.read_csv("data/demanda_promedio_real_por_mes.csv")

@st.cache_data
def cargar_mensual_pred():
    return pd.read_csv("data/demanda_promedio_predicha_por_mes.csv")

@st.cache_data
def cargar_semanal():
    return pd.read_csv("data/demanda_promedio_predicha_por_dia_semana.csv")

@st.cache_data
def cargar_anual():
    return pd.read_csv("data/demanda_promedio_anual.csv")

st.title("📈 Comparación: Demanda Real vs Predicha")

# Comparación diaria
st.subheader("📅 Por Día")
df_dia = cargar_diario()

# Confirmar columnas ya renombradas
st.write("✅ Columnas después del renombramiento:", df_dia.columns.tolist())

# Transformar de wide a long para graficar
df_dia_melt = df_dia.melt(
    id_vars="dia",
    value_vars=["demanda_real", "demanda_predicha"],
    var_name="Tipo",
    value_name="Demanda"
)

fig_dia = px.bar(df_dia_melt, x="dia", y="Demanda", color="Tipo",
                 barmode="group",
                 title="Demanda Real vs Predicha por Día de la Semana")
st.plotly_chart(fig_dia, use_container_width=True)

# Comparación mensual
st.subheader("🗓️ Promedio Mensual")

real_mensual = cargar_mensual_real()
pred_mensual = cargar_mensual_pred()

# Mostrar columnas para depuración
st.write("🟡 Columns real:", real_mensual.columns.tolist())
st.write("🔵 Columns predicha:", pred_mensual.columns.tolist())

# Merge
df_mensual = pd.merge(real_mensual, pred_mensual, on=["año", "mes"], how="inner")
st.write("🟢 Columns after merge:", df_mensual.columns.tolist())

# Renombrar columnas relevantes
df_mensual = df_mensual.rename(columns={
    "demanda_promedio_real": "Real",
    "demanda_promedio_predicha": "Predicha"
})

# Validar existencia de 'mes'
if "mes" not in df_mensual.columns:
    st.error("❌ La columna 'mes' no está disponible después del merge.")
else:
    # Preparar para gráfico
    df_mensual = df_mensual.melt(id_vars=["mes"], 
                                 value_vars=["Real", "Predicha"], 
                                 var_name="Tipo", 
                                 value_name="Demanda")

    fig_mensual = px.bar(df_mensual, x="mes", y="Demanda", color="Tipo", barmode="group",
                         title="Demanda Promedio Real vs Predicha por Mes")
    st.plotly_chart(fig_mensual, use_container_width=True)

# Comparación semanal
st.subheader("📆 Promedio por Día de la Semana")
df_semana = cargar_semanal()
df_semana = df_semana.sort_values("dia_semana")  # asegurar orden

fig_semana = px.bar(df_semana, 
                    x="dia_semana", 
                    y="demanda_promedio_predicha",
                    labels={"dia_semana": "Día de la semana", "demanda_promedio_predicha": "Demanda Predicha"},
                    title="Demanda Predicha Promedio por Día de la Semana")
st.plotly_chart(fig_semana, use_container_width=True)

# Comparación anual
st.subheader("📊 Promedio Anual")
df_anual = cargar_anual()

fig_anual = px.bar(df_anual, x="año", y="demanda_promedio_anual",
                   labels={"año": "Año", "demanda_promedio_anual": "Demanda Promedio"},
                   title="Demanda Promedio Anual")
st.plotly_chart(fig_anual, use_container_width=True)
