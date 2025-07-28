import streamlit as st
from utils import cargar_datos, entrenar_modelos, DIAS_ES
import matplotlib.pyplot as plt
import seaborn as sns

st.title('📊 Exploración Parte 2')
df = cargar_datos()
modelo_rf, _, _, _, _, _, X, _ = entrenar_modelos(df)
df['demanda_predicha'] = modelo_rf.predict(X)

tabs = st.tabs(['Real vs Predicho por Día', 'Real vs Predicho por Mes', 'Servicio más demandado'])

with tabs[0]:
    por_dia = df.groupby('dia_nombre')[['demanda_diaria', 'demanda_predicha']].sum().reindex(DIAS_ES).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    por_dia.plot(kind="bar", x="dia_nombre", ax=ax)
    st.pyplot(fig)

with tabs[1]:
    por_mes = df.groupby("mes")[["demanda_diaria", "demanda_predicha"]].sum().reindex(range(1, 13)).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    por_mes.plot(kind="bar", x="mes", ax=ax)
    st.pyplot(fig)

with tabs[2]:
    top = df.groupby(["mes_nombre", "tipo_servicio"]).agg({"demanda_diaria": "sum"}).reset_index()
    top = top.loc[top.groupby("mes_nombre")["demanda_diaria"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top, x="mes_nombre", y="demanda_diaria", hue="tipo_servicio", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)