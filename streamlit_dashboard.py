import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dashboard Notaría", layout="wide")
st.title("📊 Dashboard de Predicción de Servicios Notariales")
st.markdown("---")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_servicios_generado_demanda.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["es_finde"] = df["dia_semana"].isin([5, 6]).astype(int)
    return df

df = cargar_datos()

# ENTRENAMIENTO GLOBAL PARA EVITAR ERRORES DE NameError
df_modelo = df.dropna(subset=["demanda_diaria"])
X = df_modelo[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]]
y = df_modelo["demanda_diaria"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_dt = DecisionTreeRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)
modelo_dt.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)
pred_dt = modelo_dt.predict(X_test)

st.sidebar.header("Opciones de visualización")
seccion = st.sidebar.radio("Ir a sección:", [
    "Exploración de datos",
    "Entrenamiento y evaluación de modelos",
    "Predicción para 2024",
    "Predicción para 2025"
])

if seccion == "Exploración de datos":
    st.subheader("Distribución diaria de demanda")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df, x="fecha", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

    st.subheader("Demanda por día de la semana")
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    resumen = df.groupby("dia_semana")["demanda_diaria"].mean().reset_index()
    resumen["dia"] = resumen["dia_semana"].apply(lambda x: dias[x])
    fig, ax = plt.subplots()
    sns.barplot(data=resumen, x="dia", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

elif seccion == "Entrenamiento y evaluación de modelos":
    st.subheader("Entrenamiento de modelos")

    st.markdown("### Métricas de Random Forest")
    st.write(f"MAE: {mean_absolute_error(y_test, pred_rf):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_rf)):.2f}")
    st.write(f"R²: {r2_score(y_test, pred_rf):.2f}")

    st.markdown("### Métricas de Árbol de Decisión")
    st.write(f"MAE: {mean_absolute_error(y_test, pred_dt):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_dt)):.2f}")
    st.write(f"R²: {r2_score(y_test, pred_dt):.2f}")

elif seccion == "Predicción para 2024":
    st.subheader("🔮 Predicción para el año 2024")
    fechas_2024 = pd.date_range("2024-01-01", "2024-12-31")
    df_2024 = pd.DataFrame({"fecha": fechas_2024})
    base = df[df["fecha"].dt.year == 2023].reset_index(drop=True)
    base = base[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria"]]
    base = base.sample(n=len(df_2024), replace=True).reset_index(drop=True)

    df_2024 = pd.concat([df_2024, base], axis=1)
    df_2024["dia_semana"] = df_2024["fecha"].dt.dayofweek
    df_2024["es_finde"] = df_2024["dia_semana"].isin([5, 6]).astype(int)
    df_2024["demanda_predicha"] = modelo_rf.predict(df_2024[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]])

    df_2024["mes"] = df_2024["fecha"].dt.month
    df_2024["semana"] = df_2024["fecha"].dt.isocalendar().week

    col1, col2 = st.columns(2)
    with col1:
        mensual = df_2024.groupby("mes")["demanda_predicha"].sum().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=mensual, x="mes", y="demanda_predicha", ax=ax1)
        st.pyplot(fig1)

    with col2:
        semanal = df_2024.groupby("semana")["demanda_predicha"].sum().reset_index()
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=semanal, x="semana", y="demanda_predicha", marker="o", ax=ax2)
        st.pyplot(fig2)

    st.success(f"Demanda total estimada en 2024: {df_2024['demanda_predicha'].sum():.2f}")

elif seccion == "Predicción para 2025":
    st.subheader("🔮 Predicción para el año 2025")
    fechas_2025 = pd.date_range("2025-01-01", "2025-12-31")
    df_2025 = pd.DataFrame({"fecha": fechas_2025})
    base = df[df["fecha"].dt.year == 2023].reset_index(drop=True)
    base = base[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria"]]
    base = base.sample(n=len(df_2025), replace=True).reset_index(drop=True)

    df_2025 = pd.concat([df_2025, base], axis=1)
    df_2025["dia_semana"] = df_2025["fecha"].dt.dayofweek
    df_2025["es_finde"] = df_2025["dia_semana"].isin([5, 6]).astype(int)
    df_2025["demanda_predicha"] = modelo_rf.predict(df_2025[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]])

    df_2025["mes"] = df_2025["fecha"].dt.month
    df_2025["semana"] = df_2025["fecha"].dt.isocalendar().week

    col1, col2 = st.columns(2)
    with col1:
        mensual = df_2025.groupby("mes")["demanda_predicha"].sum().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=mensual, x="mes", y="demanda_predicha", ax=ax1)
        st.pyplot(fig1)

    with col2:
        semanal = df_2025.groupby("semana")["demanda_predicha"].sum().reset_index()
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=semanal, x="semana", y="demanda_predicha", marker="o", ax=ax2)
        st.pyplot(fig2)

    st.success(f"Demanda total estimada en 2025: {df_2025['demanda_predicha'].sum():.2f}")
