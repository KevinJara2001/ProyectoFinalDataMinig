import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Servicios más demandados", layout="wide")

# 1. Cargar CSV con encabezado correcto y limpiar nombres
@st.cache_data
def cargar_top_servicios():
    df = pd.read_csv("data/top_servicios_por_mes.csv", header=1)
    df.columns = ["mes_en", "servicio", "total", "mes"]
    return df

df_top = cargar_top_servicios()

# 2. Normalizar texto en columna "mes"
df_top["mes"] = df_top["mes"].astype(str).str.strip().str.capitalize()

# 3. Ordenar meses correctamente
orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
df_top["mes"] = pd.Categorical(df_top["mes"], categories=orden_meses, ordered=True)
df_top = df_top.sort_values("mes")

# Título
st.title("📌 Servicio Más Demandado por Mes")

# 4. Gráfico interactivo con Plotly
fig_plotly = px.bar(
    df_top, x="mes", y="total", color="servicio", barmode="group",
    title="📊 Servicio más demandado por mes",
    labels={"mes": "Mes", "total": "Demanda", "servicio": "Tipo de Servicio"}
)
st.plotly_chart(fig_plotly, use_container_width=True)

# 5. Gráfico estático con Seaborn
st.subheader("📉 Versión Estática (Seaborn)")
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_top, x="mes", y="total", hue="servicio", dodge=False)

# Mostrar valores arriba de las barras
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=8)

plt.title("Servicio más demandado por mes")
plt.xlabel("Mes")
plt.ylabel("Demanda total")
plt.xticks(rotation=45)
plt.legend(title="Tipo de Servicio", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt.gcf())
