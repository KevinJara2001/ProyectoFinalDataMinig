# reemplaza NaN por 0, en tiempo_servicio_min
df['tiempo_servicio_min'] = df['tiempo_servicio_min'].fillna(0)
df_num = df.select_dtypes(include=[np.number])

# reemplaza NaN por 0, en usuarios_diarios
df['usuarios_diarios'] = df['usuarios_diarios'].fillna(0)
df_num = df.select_dtypes(include=[np.number])

# reemplaza NaN por 0, en capacidad_maxima_diaria
df['capacidad_maxima_diaria'] = df['capacidad_maxima_diaria'].fillna(0)
df_num = df.select_dtypes(include=[np.number])

# reemplaza NaN por ninguno, en tipo_servicio
df['tipo_servicio'] = df['tipo_servicio'].fillna("ninguno")
df_num = df.select_dtypes(include=[np.number])

# cambiamos los valores NaN de hora por 00:00 de forma que el modelo lo reconozca como "sin hora valida"
# Convierte y extrae solo la hora
# 1) Imputa los NaN en 'hora' con "00:00"

# 1) Rellena los NaN de 'hora' con "00:00"
df['hora'] = df['hora'].fillna('00:00')

# 2) Convierte directamente a datetime64
df['hora_dt'] = pd.to_datetime(
    df['hora'],
    format='%H:%M',
    errors='coerce'
)

# 3) Verifica que ahora sea datetime y muestra las primeras filas
print("Tipo de dato de hora_dt:", df['hora_dt'].dtype)
df['hora_dt'].head()


# reemplaza NaN por la media, en costo_servicio
mean_costo_servicio = df['costo_servicio'].mean()
df['costo_servicio'] = df['costo_servicio'].fillna(mean_costo_servicio)
df_num = df.select_dtypes(include=[np.number])

# transformar fecha tipo string a tipo datatime para un mejor manejo
df['fecha_dt'] = pd.to_datetime(df['fecha'])

# Fecha → un entero ordinal (días desde 01-01-0001)
df['fecha_ordinal'] = df['fecha_dt'].map(pd.Timestamp.toordinal)


# Hora → minutos desde medianoche
df['hora_minutos'] = df['hora_dt'].dt.hour * 60 + df['hora_dt'].dt.minute

# transformar columna dias_semana en tipo númerico
dias_map = {
    'Lunes': 0,
    'Martes': 1,
    'Miércoles': 2,
    'Jueves': 3,
    'Viernes': 4,
    'Sábado': 5,
}

df['dia_semana_num'] = df['dia_semana'].map(dias_map)

# transformar columna "tipo" en tipo númerico
tipo_map = {
    'feriado': 0,
    'servicio': 1,
    'ausencia': 2,
    'permiso': 3,
    'vacaciones': 4,
}

df['tipo_num'] = df['tipo'].map(tipo_map)

# transformar columna "tipo_servicio" en tipo númerico
# 'ninguno' 'Actos y Contratos con Cuantía Determinada' 'Declaratoria de Propiedad Horizontal''Poderes, Procuraciones Judiciales y Contratos de Mandato''De las Sociedades''Actos Contratos y Diligencias Notariales con Tarifas Especiales''Remates Voluntarios'
# 'Actos Contratos y Dilgencia con Cuantía Indeterminada'
tipo_servicio_map = {
    'ninguno': 0,
    'Actos y Contratos con Cuantía Determinada': 1,
    'Declaratoria de Propiedad Horizontal': 2,
    'Poderes, Procuraciones Judiciales y Contratos de Mandato': 3,
    'De las Sociedades': 4,
    'Actos Contratos y Diligencias Notariales con Tarifas Especiales': 5,
    'Remates Voluntarios': 6,
    'Actos Contratos y Dilgencia con Cuantía Indeterminada': 7,
}

df['tipo_servicio_num'] = df['tipo_servicio'].map(tipo_servicio_map)

cat_cols = [
    col for col in df.select_dtypes(include=['object', 'category']).columns
    if col not in ['fecha', 'hora']
]

# Muestra solo los valores unicos
for col in cat_cols:
    print(f"Valores únicos en {col}:")
    print(df[col].unique())

df['año']  = df['fecha_dt'].dt.year
df['mes']  = df['fecha_dt'].dt.month

# Agrupa por año y mes y calcula, por ejemplo, la suma de demanda diaria
resumen = (
    df
    .groupby(['año','mes'])['demanda_diaria']
    .sum()
    .reset_index()
    .sort_values(['año','mes'])
)

# Separar variables predictoras (X) y variable objetivo (y)
# Excluye columnas no numéricas y la variable objetivo
X = df.select_dtypes(exclude=['object', "datetime64[ns]"])
X = X.drop(columns=['demanda_diaria'])
y = df["demanda_diaria"]

# se muestra el numero de filas y columnas
print("Filas y columnas de X:", X.shape)
print("Filas de Y:", y.shape[0])

X.dtypes

# Se divide en entrenamiento y prueba train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Configuración y Entrenamiento del modelo
# n_estimators -> representa todos los árboles que utiliza para entrenar el modelo
#n_jobs = -1 -> se asegura de que utilice todos los núcleos para acelerar el entrenamiento
rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)

# Predecir sobre conjunto de prueba
y_pred = rfr.predict(X_test)

# Evaluar el modelo
mae_rfr = mean_absolute_error(y_test, y_pred)
mse_rfr = mean_squared_error(y_test, y_pred)
rmse_rfr = np.sqrt(mse_rfr)
r2_rfr = r2_score(y_test, y_pred)

mae_rfr, rmse_rfr, r2_rfr
print(f"MAE: {mae_rfr:.6f}")
print(f"MSE: {mse_rfr:.6f}")
print(f"RMSE: {rmse_rfr:.6f}")
print(f"R²: {r2_rfr:.6f}")

# Configura el modelo con menos árboles y sin paralelismo excesivo
rf_model = RandomForestRegressor(
    n_estimators=50,       # reduce para acelerar
    max_depth=10,
    random_state=42,
    n_jobs=-1               # evita saturar entorno
)

# K-Fold con menos particiones, 5 para más velocidad
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Almacena métricas
mae_scores = []
mse_scores = []

# Validación cruzada manual
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    rf_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = rf_model.predict(X_val_fold)

    mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
    mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

# Resultados promedio y desviación
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"MAE promedio y desviación estándar: {avg_mae:.6f} ± {std_mae:.6f}")
print(f"MSE promedio y desviación estándar: {avg_mse:.6f} ± {std_mse:.6f}")

# Visualizar los errores por fold
plt.plot(mae_scores, marker='o', label='MAE por fold')
plt.plot(mse_scores, marker='s', label='MSE por fold')
plt.title("Errores por conjunto de datos de validación - RandomForestRegressor")
plt.xlabel("Conjunto de datos de validación")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Si conoces la media y desviación original de 'demanda_diaria':
media_original = df['demanda_diaria'].mean()     # obtiene el media real
std_original = df['demanda_diaria'].std()       # obtiene por el desviación estándar real

# Desnormalizar
y_test_real = y_test * std_original + media_original
y_pred_real = y_pred * std_original + media_original

# Graficar desnormalizado
plt.figure(figsize=(6, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.6, color='green')
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', linewidth=2)
plt.xlabel("Demanda real")
plt.ylabel("Demanda predicha")
plt.title("Predicción vs Valor Real")
plt.grid(True)
plt.tight_layout()
plt.show()


errores = y_test - y_pred

# Histograma de errores
plt.hist(errores, bins=30, edgecolor='black')
plt.title("Distribución de errores")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Errores vs predicción
plt.scatter(y_pred, errores, alpha=0.5, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.title("Errores vs Predicción")
plt.xlabel("Predicción")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# Definir el modelo base con parámetros razonables
dt = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,               # Profundidad controlada
    min_samples_split=5,       # Evita divisiones excesivas
    min_samples_leaf=2         # Hojas con mínimo 2 muestras
)

# Entrenamiento
dt.fit(X_train, y_train)

# Predicción y métricas
y_pred = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred)
mae_dt = mean_absolute_error(y_test, y_pred)
rmse_dt = np.sqrt(mse_dt)
r2_dt  = r2_score(y_test, y_pred)

print(f"MAE: {mae_dt:.6f}")
print(f"MSE: {mse_dt:.6f}")
print(f"RMSE:{rmse_dt:.6f}")
print(f"R²:  {r2_dt:.6f}")

# Visualización del árbol (primeros 3 niveles)
plt.figure(figsize=(16, 10))
plot_tree(
    dt,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10
)
plt.title("Árbol de Decisión (Visualización hasta nivel 3)")
plt.show()

# Búsqueda de mejores hiperparámetros con GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # también podrías usar 'r2' o 'neg_mean_squared_error'
    cv=5,
    verbose=1
)
# entrenamiento
grid_dt.fit(X_train, y_train)

# Resultados de la búsqueda
print("Mejores parámetros encontrados:", grid_dt.best_params_)
print("Mejor MAE en validación cruzada:", -grid_dt.best_score_)

# Definir el modelo
dt_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,
    min_samples_split=2,  # ajustar si es necesario
    min_samples_leaf=1
)

# K-Fold configuration, 5 subconjuntos de datos de validación
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Almacenar métricas
mae_scores = []
mse_scores = []

# Validación cruzada
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    dt_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = dt_model.predict(X_val_fold)

    mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
    mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

# Resultados agregados
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"MAE promedio y desviacion estandar: {avg_mae:.6f} ± {std_mae:.6f}")
print(f"MSE promedio y desviacion estandar: {avg_mse:.6f} ± {std_mse:.6f}")

# Visualización de errores por fold
plt.plot(mae_scores, marker='o', label='MAE por fold')
plt.plot(mse_scores, marker='s', label='MSE por fold')
plt.title("Errores por subconjuntos de datos de validación - Árbol de Decisión (max_depth=5)")
plt.xlabel("Fold")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfecto")
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Valor Real - Árbol de Decisión")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

errores = y_test - y_pred

# Histograma de errores
plt.hist(errores, bins=30, edgecolor='black')
plt.title("Distribución de errores")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Errores vs predicción
plt.scatter(y_pred, errores, alpha=0.5, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.title("Errores vs Predicción")
plt.xlabel("Predicción")
plt.ylabel("Error")
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Diccionario vacío donde se irán agregando resultados
resultados = []

# === EJEMPLO: Agrega resultados luego de entrenar cada modelo ===

# Árbol de Decisión
resultados.append({
    "Modelo": "Árbol de Decisión",
    "MAE": mae_dt,
    "RMSE": rmse_dt,
    "MSE": mse_dt,
    "R²": r2_dt
})

# Random Forest
resultados.append({
    "Modelo": "Random Forest",
    "MAE": mae_rfr,
    "RMSE": rmse_rfr,
    "MSE": mse_rfr,
    "R²": r2_rfr
})

# === CONVERTIR A DATAFRAME ===
df_resultados = pd.DataFrame(resultados)

# Mostrar tabla en consola
print("Comparación de Modelos:")
print(df_resultados)

# === GRAFICAR COMPARACIÓN ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# MAE
sns.barplot(x="Modelo", y="MAE", data=df_resultados, ax=axs[0, 0], palette="Blues")
axs[0, 0].set_title("MAE - Error Absoluto Medio")
axs[0, 0].set_ylabel("MAE")

# RMSE
sns.barplot(x="Modelo", y="RMSE", data=df_resultados, ax=axs[0, 1], palette="Greens")
axs[0, 1].set_title("RMSE - Raíz del Error Cuadrático Medio")
axs[0, 1].set_ylabel("RMSE")

# MSE
sns.barplot(x="Modelo", y="MSE", data=df_resultados, ax=axs[1, 0], palette="Oranges")
axs[1, 0].set_title("MSE - Error Cuadrático Medio")
axs[1, 0].set_ylabel("MSE")

# R²
sns.barplot(x="Modelo", y="R²", data=df_resultados, ax=axs[1, 1], palette="Purples")
axs[1, 1].set_title("R² - Coeficiente de Determinación")
axs[1, 1].set_ylabel("R²")

plt.suptitle("Comparación de Desempeño entre Modelos", fontsize=16)
plt.tight_layout()
plt.show()

# Guardar las columnas antes de escalar
columnas = X.columns

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Concatenar escalados
X_scaled = np.concatenate((X_train_scaled, X_test_scaled), axis=0)

# Crear DataFrame con las columnas correctas
X_scaled_df = pd.DataFrame(X_scaled, columns=columnas)

# Guardar a CSV
X_scaled_df.to_csv("dataset_servicios_normalizado.csv", index=False)


# Generar predicciones para todo el dataset
y_pred_all = rfr.predict(X)

# Añadir predicciones al DataFrame original
df['demanda_predicha'] = y_pred_all

df['mes'] = df['fecha_dt'].dt.month

# Agrupar por mes
por_mes = df.groupby('mes')[['demanda_diaria', 'demanda_predicha']].sum().reset_index()

# Visualizar
por_mes.plot(x='mes', kind='bar', figsize=(10,5), title='Demanda Real vs Predicha por Mes')
plt.ylabel('Demanda total')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

# Día de la semana ya numérico como 'dia_semana_num'
por_dia = df.groupby('dia_semana_num')[['demanda_diaria', 'demanda_predicha']].sum().reset_index()

# Mapear número a nombre
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
por_dia['dia'] = por_dia['dia_semana_num'].map(dict(zip(range(6), dias)))

# Visualizar
por_dia.plot(x='dia', kind='bar', figsize=(10,5), title='Demanda Real vs Predicha por Día de la Semana')
plt.ylabel('Demanda total')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()


# Agrupación
df_mensual = (
    df.groupby(['año', 'mes'])[['demanda_diaria', 'demanda_predicha']]
    .mean()
    .reset_index()
    .sort_values(['año', 'mes'])
)

# Mostrar tabla
print(df_mensual)


# Gráfico de líneas
plt.figure(figsize=(12,6))
for año in df_mensual['año'].unique():
    datos_año = df_mensual[df_mensual['año'] == año]
    plt.plot(datos_año['mes'], datos_año['demanda_diaria'], label=f'Real {año}', marker='o')
    plt.plot(datos_año['mes'], datos_año['demanda_predicha'], label=f'Predicha {año}', marker='x')

plt.title('Demanda Promedio Real vs Predicha por Mes y Año')
plt.xlabel('Mes')
plt.ylabel('Demanda Promedio')
plt.xticks(range(1,13))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Asegura que la columna 'dia_semana' exista
if 'dia_semana' not in df.columns:
    df['dia_semana'] = df['fecha_dt'].dt.day_name(locale='es_ES')

# Calcula el promedio de demanda predicha por día de la semana
daily_predictions = (
    df.groupby('dia_semana')[['demanda_predicha']]
    .mean()
    .reset_index()
    .rename(columns={'demanda_predicha': 'demanda_promedio_predicha'})
)

# Ordena los días correctamente
orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
daily_predictions['dia_semana'] = pd.Categorical(
    daily_predictions['dia_semana'],
    categories=orden_dias,
    ordered=True
)
daily_predictions = daily_predictions.sort_values('dia_semana')

# Muestra resultados en tabla
print(daily_predictions)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=daily_predictions,
    x='dia_semana',
    y='demanda_promedio_predicha',
    palette='viridis'
)
plt.title("Demanda Promedio Predicha por Día de la Semana", fontsize=14)
plt.ylabel("Demanda Promedio Predicha", fontsize=12)
plt.xlabel("Día de la Semana", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Agrupar y calcular la demanda promedio predicha
monthly_predictions = (
    df.groupby(['año', 'mes'])['demanda_predicha']
    .mean()
    .reset_index()
    .rename(columns={'demanda_predicha': 'demanda_promedio_predicha'})
)

# Crear columna de fecha real desde columnas 'año' y 'mes'
monthly_predictions['fecha'] = pd.to_datetime({
    'year': monthly_predictions['año'],
    'month': monthly_predictions['mes'],
    'day': 1  # Día por defecto para representar el mes
})


plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_predictions, x='fecha', y='demanda_promedio_predicha', marker='o', color='tab:blue')
plt.title('Demanda Promedio Predicha por Mes')
plt.xlabel('Fecha')
plt.ylabel('Demanda Promedio Predicha')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Asegura de que las columnas 'fecha_dt', 'año' y 'mes' existen
if 'fecha_dt' not in df.columns:
    df['fecha_dt'] = pd.to_datetime(df['fecha'])
df['año'] = df['fecha_dt'].dt.year
df['mes'] = df['fecha_dt'].dt.month

# Agrupar por año y mes, calcular demanda promedio real
monthly_real = (
    df.groupby(['año', 'mes'])['demanda_diaria']
    .mean()
    .reset_index()
    .rename(columns={'demanda_diaria': 'demanda_promedio_real'})
)

# Crear columna fecha (primer día del mes) para eje X del gráfico
monthly_real['fecha'] = pd.to_datetime({
    'year': monthly_real['año'],
    'month': monthly_real['mes'],
    'day': 1
})

# 4. Visualización: Gráfico de línea de la demanda mensual promedio
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=monthly_real,
    x='fecha',
    y='demanda_promedio_real',
    marker='o',
    linewidth=2,
    color='teal'
)
plt.title("Demanda Promedio Real por Mes", fontsize=14)
plt.xlabel("Mes")
plt.ylabel("Demanda Promedio")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Asegúrate de tener la columna 'año' extraída de la fecha
if 'año' not in df.columns:
    df['año'] = df['fecha_dt'].dt.year

# Agrupar por año y calcular la demanda promedio
yearly_avg = (
    df.groupby('año')['demanda_diaria']
    .mean()
    .reset_index()
    .rename(columns={'demanda_diaria': 'demanda_promedio_anual'})
)

plt.figure(figsize=(8, 5))
sns.barplot(data=yearly_avg, x='año', y='demanda_promedio_anual', palette='crest')
plt.title("Demanda Promedio Anual")
plt.xlabel("Año")
plt.ylabel("Demanda Promedio")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Filtrar filas válidas
df_filtrado = df.dropna(subset=['tipo_servicio', 'demanda_diaria'])

# Usa la columna ya convertida 'fecha_dt'
df_filtrado['mes'] = df_filtrado['fecha_dt'].dt.month_name()

# Agrupar por mes y tipo de servicio, sumando demanda
agrupado = df_filtrado.groupby(['mes', 'tipo_servicio'])['demanda_diaria'].sum().reset_index()

# Ordenar por demanda dentro de cada mes
agrupado_ordenado = agrupado.sort_values(['mes', 'demanda_diaria'], ascending=[True, False])

# Obtener servicio con más demanda por mes
top_servicios_mes = agrupado_ordenado.groupby('mes').first().reset_index()

# Mostrar tabla
print("Servicios más utilizados por mes:")
display(top_servicios_mes)

# Servicio más utilizado del año
servicio_top_anual = df_filtrado.groupby('tipo_servicio')['demanda_diaria'].sum().idxmax()
print(f"\nServicio más utilizado en el año 2024: {servicio_top_anual}")

import matplotlib.pyplot as plt
import seaborn as sns

# Mapeo de meses de inglés a español
meses_es = {
    "January": "Enero", "February": "Febrero", "March": "Marzo", "April": "Abril",
    "May": "Mayo", "June": "Junio", "July": "Julio", "August": "Agosto",
    "September": "Septiembre", "October": "Octubre", "November": "Noviembre", "December": "Diciembre"
}

# Aplicar mapeo
top_servicios_mes['mes_es'] = top_servicios_mes['mes'].map(meses_es)

# Orden cronológico
orden_meses_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                  "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
top_servicios_mes['mes_es'] = pd.Categorical(top_servicios_mes['mes_es'], categories=orden_meses_es, ordered=True)
top_servicios_mes = top_servicios_mes.sort_values("mes_es")

# Gráfico de barras con meses
plt.figure(figsize=(12, 6))
sns.barplot(data=top_servicios_mes, x='mes_es', y='demanda_diaria', hue='tipo_servicio', dodge=False)
plt.title('Servicio más demandado por mes')
plt.xlabel('Mes')
plt.ylabel('Demanda total')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Servicio', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 📌 PASO 1: Cargar datos reales y preparar 2023
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Cargar archivo desde la carpeta izquierda
df = pd.read_csv("/content/dataset_servicios_generado_demanda.csv")
df["fecha"] = pd.to_datetime(df["fecha"])

# Filtrar solo 2023 como base de comportamiento
df_2023 = df[(df["fecha"].dt.year == 2023)].copy()
df_2023 = df_2023.dropna(subset=["demanda_diaria"])

# 📌 PASO 2: Generar calendario 2024
fechas_2024 = pd.date_range("2024-01-01", "2024-12-31", freq="D")
df_2024 = pd.DataFrame({"fecha": fechas_2024})

# Reutilizar el patrón de días de 2023 (asumiendo mismo orden de calendario)
df_2023_pattern = df_2023[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria"]].reset_index(drop=True)
if len(df_2023_pattern) > len(df_2024):
    df_2023_pattern = df_2023_pattern.iloc[:len(df_2024)]
elif len(df_2023_pattern) < len(df_2024):
    df_2023_pattern = df_2023_pattern.sample(len(df_2024), replace=True).reset_index(drop=True)

# Simular 2024 con variación
np.random.seed(42)
ruido = lambda serie, std: serie + np.random.normal(0, std, len(serie))

df_2024["usuarios_diarios"] = ruido(df_2023_pattern["usuarios_diarios"], 2).clip(lower=0)
df_2024["tiempo_servicio_min"] = ruido(df_2023_pattern["tiempo_servicio_min"], 1).clip(lower=0)
df_2024["costo_servicio"] = ruido(df_2023_pattern["costo_servicio"], 50).clip(lower=0)
df_2024["capacidad_maxima_diaria"] = ruido(df_2023_pattern["capacidad_maxima_diaria"], 1).clip(lower=0)

# 📌 PASO 3: Entrenar modelo con todo el histórico
df["dia_semana"] = df["fecha"].dt.weekday
df["es_finde"] = df["dia_semana"].isin([5, 6]).astype(int)

X = df[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]].fillna(df.mean(numeric_only=True))
y = df["demanda_diaria"].fillna(df["demanda_diaria"].mean())

modelo = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42)
modelo.fit(X, y)

# 📌 PASO 4: Predicción para 2024
df_2024["dia_semana"] = df_2024["fecha"].dt.weekday
df_2024["es_finde"] = df_2024["dia_semana"].isin([5, 6]).astype(int)

X_2024 = df_2024[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]]
df_2024["demanda_predicha"] = modelo.predict(X_2024)

# 📌 PASO 5: Agregaciones
df_2024["mes"] = df_2024["fecha"].dt.month
df_2024["semana"] = df_2024["fecha"].dt.isocalendar().week

mensual = df_2024.groupby("mes")["demanda_predicha"].sum().reset_index()
semanal = df_2024.groupby("semana")["demanda_predicha"].sum().reset_index()

# 📊 PASO 6: Visualización
plt.figure(figsize=(10, 4))
sns.barplot(data=mensual, x="mes", y="demanda_predicha")
plt.title("📆 Demanda mensual predicha - 2024")
plt.xlabel("Mes")
plt.ylabel("Demanda total")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
sns.lineplot(data=semanal, x="semana", y="demanda_predicha", marker="o")
plt.title("📆 Demanda semanal predicha - 2024")
plt.xlabel("Semana")
plt.ylabel("Demanda total")
plt.tight_layout()
plt.grid()
plt.show()

# 📌 Total
total = df_2024["demanda_predicha"].sum()
print(f"📌 Demanda total estimada en 2024: {total:.2f}")


# 📌 PASO 1: Crear fechas de 2025
fechas_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
df_2025 = pd.DataFrame({"fecha": fechas_2025})

# Reutilizar el patrón de días de 2023
df_2023_pattern = df_2023[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria"]].reset_index(drop=True)
if len(df_2023_pattern) > len(df_2025):
    df_2023_pattern = df_2023_pattern.iloc[:len(df_2025)]
elif len(df_2023_pattern) < len(df_2025):
    df_2023_pattern = df_2023_pattern.sample(len(df_2025), replace=True).reset_index(drop=True)

# Simulación con ruido
np.random.seed(123)
df_2025["usuarios_diarios"] = ruido(df_2023_pattern["usuarios_diarios"], 2).clip(lower=0)
df_2025["tiempo_servicio_min"] = ruido(df_2023_pattern["tiempo_servicio_min"], 1).clip(lower=0)
df_2025["costo_servicio"] = ruido(df_2023_pattern["costo_servicio"], 50).clip(lower=0)
df_2025["capacidad_maxima_diaria"] = ruido(df_2023_pattern["capacidad_maxima_diaria"], 1).clip(lower=0)

# Predicción
df_2025["dia_semana"] = df_2025["fecha"].dt.weekday
df_2025["es_finde"] = df_2025["dia_semana"].isin([5, 6]).astype(int)

X_2025 = df_2025[["usuarios_diarios", "tiempo_servicio_min", "costo_servicio", "capacidad_maxima_diaria", "es_finde"]]
df_2025["demanda_predicha"] = modelo.predict(X_2025)

# Agregaciones
df_2025["mes"] = df_2025["fecha"].dt.month
df_2025["semana"] = df_2025["fecha"].dt.isocalendar().week

mensual_2025 = df_2025.groupby("mes")["demanda_predicha"].sum().reset_index()
semanal_2025 = df_2025.groupby("semana")["demanda_predicha"].sum().reset_index()

# Gráficos
plt.figure(figsize=(10, 4))
sns.barplot(data=mensual_2025, x="mes", y="demanda_predicha")
plt.title("📆 Demanda mensual predicha - 2025")
plt.xlabel("Mes")
plt.ylabel("Demanda total")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
sns.lineplot(data=semanal_2025, x="semana", y="demanda_predicha", marker="o")
plt.title("📆 Demanda semanal predicha - 2025")
plt.xlabel("Semana")
plt.ylabel("Demanda total")
plt.tight_layout()
plt.grid()
plt.show()

# Total anual
total_2025 = df_2025["demanda_predicha"].sum()
print(f"📌 Demanda total estimada en 2025: {total_2025:.2f}")

# Imputar NaN restantes con la media (en variables numéricas) para evitar error
X = X.fillna(X.mean())

# Volver a ajustar y graficar
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importancias = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importancias, y=features)
plt.title("Importancia de Variables - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()
plt.grid(True)
plt.show()

# Validación cruzada nuevamente
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    mae_scores.append(mean_absolute_error(y_val, y_pred))
    mse_scores.append(mean_squared_error(y_val, y_pred))

plt.figure(figsize=(10, 5))
plt.plot(mae_scores, marker='o', label='MAE por fold')
plt.plot(mse_scores, marker='s', label='MSE por fold')
plt.title("Errores por Fold - Validación Cruzada")
plt.xlabel("Fold")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mapa de correlaciones actualizado
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Mapa de Correlaciones (post procesamiento)")
plt.tight_layout()
plt.show()