import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

MESES_ES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
DIAS_ES = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

def cargar_datos():
    df = pd.read_csv("dataset_servicios_normalizado.csv")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["es_finde"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["mes"] = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year
    df["dia_nombre"] = df["dia_semana"].apply(lambda x: DIAS_ES[x])
    df["mes_nombre"] = df["mes"].apply(lambda x: MESES_ES[x-1])
    return df

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