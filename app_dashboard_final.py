
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

st.set_page_config(layout="wide", page_title="Dashboard Completo de Demanda Notarial")
st.title("📊 Dashboard Completo de Predicción de Demanda Notarial")

# Sidebar
st.sidebar.title("🔧 Opciones del Dashboard")

dataset_opcion = st.sidebar.selectbox("📁 Dataset", ["Original", "Normalizado"])
df = pd.read_csv("dataset_servicios_generado_demanda.csv")

if 'fecha' in df.columns:
    df['fecha_dt'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['año'] = df['fecha_dt'].dt.year
    df['mes'] = df['fecha_dt'].dt.month
    df['dia_semana'] = df['fecha_dt'].dt.day_name()

# Filtros
if 'año' in df.columns:
    años = sorted(df['año'].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("📅 Año", años, index=len(años)-1)
    df = df[df['año'] == año_seleccionado]

if 'tipo_servicio' in df.columns:
    tipos = df['tipo_servicio'].dropna().unique()
    tipo_sel = st.sidebar.multiselect("📌 Tipo de servicio", tipos, default=list(tipos))
    df = df[df['tipo_servicio'].isin(tipo_sel)]

pagina = st.sidebar.radio("🧭 Navegar por:", [
    "📅 Predicción Futura", "Modelos", "Visualización", "Importancia", "Comparación"
])

elif pagina == "Modelos":
    st.header("🧠 Entrenamiento de Modelos")
    columnas_usar = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'demanda_diaria' not in columnas_usar:
        st.warning("No se encontró la columna 'demanda_diaria'")
    else:
        X = df[[c for c in columnas_usar if c != 'demanda_diaria']].fillna(0)
        y = df['demanda_diaria']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.success("✅ Modelo Random Forest entrenado")
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

elif pagina == "Visualización":
    st.header("📆 Predicción por Mes y Día")
    if 'demanda_diaria' in df.columns:
        df['prediccion'] = RandomForestRegressor(n_estimators=100, random_state=42).fit(
            df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0),
            df['demanda_diaria']
        ).predict(df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0))

        mensual = df.groupby('mes')[['demanda_diaria', 'prediccion']].sum()
        st.subheader("📈 Demanda mensual real vs predicha")
        st.line_chart(mensual)

        if 'dia_semana' in df.columns:
            dias = df.groupby('dia_semana')[['demanda_diaria', 'prediccion']].sum()
            st.subheader("📅 Demanda por día de la semana")
            st.bar_chart(dias)

elif pagina == "Importancia":
    st.header("🔍 Importancia de Variables")
    if 'demanda_diaria' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0)
        y = df['demanda_diaria']
        modelo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        importancias = modelo.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=importancias, y=X.columns, ax=ax)
        ax.set_title("Importancia de variables")
        st.pyplot(fig)

elif pagina == "Comparación":
    st.header("📉 Comparación Real vs Predicho")
    if 'demanda_diaria' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0)
        y = df['demanda_diaria']
        modelo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        pred = modelo.predict(X)
        df['pred'] = pred
        st.line_chart(df[['demanda_diaria', 'pred']])


elif pagina == "📅 Predicción Futura":
    st.header("🔮 Predicción de Demanda para Años Futuros")

    # Entrenar modelo con todo el dataset histórico
    if 'demanda_diaria' in df.columns and 'fecha_dt' in df.columns:
        X = df.select_dtypes(include=[np.number]).drop(columns=['demanda_diaria'], errors='ignore').fillna(0)
        y = df['demanda_diaria']
        modelo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

        # Año futuro a simular
        anio_futuro = st.selectbox("Selecciona un año a predecir:", [2024, 2025, 2026])
        meses = list(range(1, 13))
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
        tipo_servicios = df['tipo_servicio'].dropna().unique().tolist()

        # Crear dataset futuro simulado
        datos_futuros = []
        for mes in meses:
            for dia in range(1, 29):  # hasta 28 para evitar problemas con febrero
                fecha = datetime.date(anio_futuro, mes, dia)
                dia_semana = fecha.strftime("%A")
                if dia_semana == "Sunday":
                    continue  # omitimos domingos

                datos_futuros.append({
                    'fecha': fecha,
                    'año': anio_futuro,
                    'mes': mes,
                    'dia_semana': dia_semana,
                    'capacidad_maxima_diaria': df['capacidad_maxima_diaria'].mean(),
                    'costo_servicio': df['costo_servicio'].mean(),
                    'usuarios_diarios': df['usuarios_diarios'].mean(),
                    'tiempo_servicio_min': df['tiempo_servicio_min'].mean(),
                    'tipo_num': 1,
                    'tipo_servicio_num': 1,
                    'hora_minutos': 600  # 10:00 AM como referencia
                })

        df_futuro = pd.DataFrame(datos_futuros)

        # Preparar para predicción
        columnas_modelo = modelo.feature_names_in_
        for col in columnas_modelo:
            if col not in df_futuro.columns:
                df_futuro[col] = 0
        X_futuro = df_futuro[columnas_modelo].fillna(0)

        # Predecir
        predicciones = modelo.predict(X_futuro)
        df_futuro['prediccion'] = predicciones

        # Agregar columna fecha para visualizar
        df_futuro['fecha'] = pd.to_datetime(df_futuro['fecha'])
        df_futuro['mes'] = df_futuro['fecha'].dt.month

        st.subheader(f"📈 Demanda mensual predicha para {anio_futuro}")
        resumen_mes = df_futuro.groupby('mes')['prediccion'].sum()
        st.bar_chart(resumen_mes)

        st.subheader(f"🗓️ Línea temporal diaria - {anio_futuro}")
        df_futuro_ordenado = df_futuro.sort_values('fecha')
        st.line_chart(df_futuro_ordenado.set_index('fecha')['prediccion'])
