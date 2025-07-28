
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Dashboard de Demanda Notarial")

st.title("📊 Dashboard de Predicción de Demanda Notarial")

# Sidebar
st.sidebar.title("🔧 Opciones del Dashboard")

# Dataset
dataset_opcion = st.sidebar.selectbox("📁 Dataset", ["Original", "Normalizado"])
df = pd.read_csv("dataset_servicios_generado_demanda.csv")
if 'fecha' in df.columns:
    df['fecha_dt'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['año'] = df['fecha_dt'].dt.year
    df['mes'] = df['fecha_dt'].dt.month
    df['dia_semana'] = df['fecha_dt'].dt.day_name()

# Filtros dinámicos
if 'año' in df.columns:
    años = sorted(df['año'].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("📅 Año", años, index=len(años)-1)
    df = df[df['año'] == año_seleccionado]

if 'tipo_servicio' in df.columns:
    tipos = df['tipo_servicio'].dropna().unique()
    tipo_sel = st.sidebar.multiselect("📌 Tipo de servicio", tipos, default=list(tipos))
    df = df[df['tipo_servicio'].isin(tipo_sel)]

# Navegación por páginas
pagina = st.sidebar.radio("🧭 Navegar por:", [
    "Exploración de Datos",
    "Entrenamiento del Modelo",
    "Comparación y Predicción"
])

if pagina == "Exploración de Datos":
    st.header("🔍 Exploración de Datos")
    st.dataframe(df.head())

    st.subheader("Distribución de Variables Numéricas")
    st.bar_chart(df.select_dtypes(include=np.number))

    if 'demanda_diaria' in df.columns:
        st.subheader("Demanda diaria por mes")
        demanda_mes = df.groupby('mes')['demanda_diaria'].sum()
        st.line_chart(demanda_mes)

elif pagina == "Entrenamiento del Modelo":
    st.header("🧠 Entrenamiento de Random Forest")

    # Selección simple de columnas
    columnas_usar = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'demanda_diaria' not in columnas_usar:
        st.warning("No se encontró la columna 'demanda_diaria' para entrenar el modelo.")
    else:
        X = df[[c for c in columnas_usar if c != 'demanda_diaria']].fillna(0)
        y = df['demanda_diaria']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Modelo entrenado correctamente")

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        col3.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

        st.subheader("Importancia de Variables")
        importancias = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(X.columns, importancias)
        st.pyplot(fig)

elif pagina == "Comparación y Predicción":
    st.header("📈 Comparación Real vs Predicho")

    if 'demanda_diaria' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0)
        y = df['demanda_diaria']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        df['prediccion'] = pred
        st.line_chart(df[['demanda_diaria', 'prediccion']])
