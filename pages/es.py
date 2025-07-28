import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

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

@st.cache_resource
def entrenar_modelo(df):
    X = df[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]].dropna()
    y = df["demanda_diaria"].dropna()
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)
    return modelo_rf

df = cargar_datos()
modelo_rf = entrenar_modelo(df)
df["demanda_predicha"] = modelo_rf.predict(df[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]])

st.header("Exploración de Datos - Parte 2")
tabs = st.tabs(["Real vs Predicho por Día", "Real vs Predicho por Mes", "Servicio más demandado"])

with tabs[0]:
    st.markdown("### Comparación real vs predicho por día")
    por_dia = df.groupby("dia_nombre")[["demanda_diaria", "demanda_predicha"]].sum().reindex(DIAS_ES).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
    por_dia.plot(kind="bar", x="dia_nombre", ax=ax)
    st.pyplot(fig)

with tabs[1]:
    st.markdown("### Comparación real vs predicho por mes")
    por_mes = df.groupby("mes")[["demanda_diaria", "demanda_predicha"]].sum().reindex(range(1, 13)).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
    por_mes.plot(kind="bar", x="mes", ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.markdown("### Servicio más demandado por mes")
    top = df.groupby(["mes_nombre", "tipo_servicio"]).agg({"demanda_diaria": "sum"}).reset_index()
    top = top.loc[top.groupby("mes_nombre")["demanda_diaria"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
    sns.barplot(data=top, x="mes_nombre", y="demanda_diaria", hue="tipo_servicio", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)