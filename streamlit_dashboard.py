import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gc

st.set_page_config(page_title="Dashboard Notaría", layout="wide")
st.title("📊 Dashboard de Predicción de Servicios Notariales")
st.markdown("---")

MESES_ES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
DIAS_ES = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_servicios_generado_demanda.csv")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["es_finde"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["mes"] = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["dia_nombre"] = df["dia_semana"].apply(lambda x: DIAS_ES[x])
    df["mes_nombre"] = df["mes"].apply(lambda x: MESES_ES[x-1])
    return df

@st.cache_resource
def entrenar_modelos(df):
    df_modelo = df.dropna(subset=["demanda_diaria"])
    X = df_modelo[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]]
    y = df_modelo["demanda_diaria"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    modelo_rf.fit(X_train, y_train)
    modelo_dt.fit(X_train, y_train)
    pred_rf = modelo_rf.predict(X_test)
    pred_dt = modelo_dt.predict(X_test)
    return modelo_rf, modelo_dt, X_test, y_test, pred_rf, pred_dt, X, y

df = cargar_datos()
modelo_rf, modelo_dt, X_test, y_test, pred_rf, pred_dt, X, y = entrenar_modelos(df)

st.sidebar.header("Opciones de visualización")
seccion = st.sidebar.radio("Ir a sección:", [
    "Exploración de datos",
    "Entrenamiento y evaluación de modelos",
    "Predicción para 2024",
    "Predicción para 2025"
])

if seccion == "Exploración de datos":
    st.subheader("Distribución diaria de demanda")
    fig, ax = plt.subplots(figsize=(10, 3), dpi=72)
    sns.lineplot(data=df, x="fecha", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

    st.subheader("Demanda por día de la semana")
    resumen = df.groupby("dia_nombre")["demanda_diaria"].mean().reindex(DIAS_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
    sns.barplot(data=resumen, x="dia_nombre", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

    st.subheader("Demanda mensual promedio")
    resumen_m = df.groupby("mes_nombre")["demanda_diaria"].mean().reindex(MESES_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3), dpi=72)
    sns.barplot(data=resumen_m, x="mes_nombre", y="demanda_diaria", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif seccion == "Entrenamiento y evaluación de modelos":
    st.subheader("Entrenamiento de modelos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Random Forest")
        st.write(f"MAE: {mean_absolute_error(y_test, pred_rf):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_rf)):.2f}")
        st.write(f"R²: {r2_score(y_test, pred_rf):.2f}")

    with col2:
        st.markdown("### Árbol de Decisión")
        st.write(f"MAE: {mean_absolute_error(y_test, pred_dt):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_dt)):.2f}")
        st.write(f"R²: {r2_score(y_test, pred_dt):.2f}")

    st.subheader("Comparación gráfica de modelos")
    metrics_df = pd.DataFrame({
        "Modelo": ["Árbol de Decisión", "Random Forest"],
        "MAE": [mean_absolute_error(y_test, pred_dt), mean_absolute_error(y_test, pred_rf)],
        "RMSE": [np.sqrt(mean_squared_error(y_test, pred_dt)), np.sqrt(mean_squared_error(y_test, pred_rf))],
        "MSE": [mean_squared_error(y_test, pred_dt), mean_squared_error(y_test, pred_rf)],
        "R2": [r2_score(y_test, pred_dt), r2_score(y_test, pred_rf)]
    })

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=72)
    sns.barplot(data=metrics_df, x="Modelo", y="MAE", ax=axs[0, 0])
    axs[0, 0].set_title("MAE")
    sns.barplot(data=metrics_df, x="Modelo", y="RMSE", ax=axs[0, 1])
    axs[0, 1].set_title("RMSE")
    sns.barplot(data=metrics_df, x="Modelo", y="MSE", ax=axs[1, 0])
    axs[1, 0].set_title("MSE")
    sns.barplot(data=metrics_df, x="Modelo", y="R2", ax=axs[1, 1])
    axs[1, 1].set_title("R²")
    st.pyplot(fig)

elif "Predicción para" in seccion:
    year = 2024 if "2024" in seccion else 2025
    st.subheader(f"🔮 Predicción para el año {year}")

    fechas = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    df_future = pd.DataFrame({"fecha": fechas})
    base = df[df["anio"] == 2023][["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria"]].sample(n=len(df_future), replace=True).reset_index(drop=True)
    df_future = pd.concat([df_future, base], axis=1)
    df_future["dia_semana"] = df_future["fecha"].dt.dayofweek
    df_future["es_finde"] = df_future["dia_semana"].isin([5, 6]).astype(int)
    df_future["demanda_predicha"] = modelo_rf.predict(df_future[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]])
    df_future["mes_nombre"] = df_future["fecha"].dt.month.apply(lambda x: MESES_ES[x-1])
    df_future["semana"] = df_future["fecha"].dt.isocalendar().week

    col1, col2 = st.columns(2)
    with col1:
        mensual = df_future.groupby("mes_nombre")["demanda_predicha"].sum().reindex(MESES_ES).reset_index()
        fig1, ax1 = plt.subplots(figsize=(8, 3), dpi=72)
        sns.barplot(data=mensual, x="mes_nombre", y="demanda_predicha", ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

    with col2:
        semanal = df_future.groupby("semana")["demanda_predicha"].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 3), dpi=72)
        sns.lineplot(data=semanal, x="semana", y="demanda_predicha", marker="o", ax=ax2)
        st.pyplot(fig2)

    st.success(f"Demanda total estimada en {year}: {df_future['demanda_predicha'].sum():.2f}")
    del df_future; gc.collect()
