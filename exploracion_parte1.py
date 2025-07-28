import streamlit as st
from utils import cargar_datos, MESES_ES, DIAS_ES
import matplotlib.pyplot as plt
import seaborn as sns

df = cargar_datos()
st.title('📊 Exploración Parte 1')
tabs = st.tabs(['Demanda diaria', 'Por día de semana', 'Por mes', 'Demanda anual'])

with tabs[0]:
    st.markdown('### Distribución diaria de demanda')
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x='fecha', y='demanda_diaria', ax=ax)
    st.pyplot(fig)

with tabs[1]:
    st.markdown('### Demanda por día de la semana')
    resumen = df.groupby("dia_nombre")["demanda_diaria"].mean().reindex(DIAS_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=resumen, x="dia_nombre", y="demanda_diaria", ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.markdown('### Demanda mensual promedio')
    resumen_m = df.groupby("mes_nombre")["demanda_diaria"].mean().reindex(MESES_ES).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=resumen_m, x="mes_nombre", y="demanda_diaria", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

with tabs[3]:
    st.markdown('### Demanda promedio anual')
    anual = df.groupby("anio")["demanda_diaria"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=anual, x="anio", y="demanda_diaria", ax=ax)
    st.pyplot(fig)