# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image




st.title("Interrupción Legal del Embarazo en México")
st.markdown("Desde hace muchos años el catolicismo se encuentra muy presente en la sociedad mexicana y la Iglesia juega un rol político importante, aún así México ha dado pasos notables en la liberalización de las leyes de aborto. En abril de 2007 la Ciudad de México aprobó la despenalización y comenzó a ofrecer el servicio en hospitales públicos e instituciones de salud. Todo esto fue posible gracias al impulso del movimiento feminista, la polarización electoral y la presencia del Partido de la Revolución Democrática (PRD) en la CDMX. Este logro es parte de un largo camino que ubica a México a la vanguardia de América Latina en el tratamiento humanitario del aborto. La interrupción legal del embarazo es un tema de salud pública, no es un asunto que involucre la moral ni lo religioso. A 14 años de haber entrado en vigor las reformas al Código Penal de la Ciudad de México, que permiten el aborto hasta las 12 semanas de gestación, más 76 mil 744 mujeres han recurrido a este procedimiento.")
st.sidebar.title("Cifras sobre el aborto en México")

datos = pd.read_csv('interrupcion-legal-del-embarazo.csv')
df= pd.DataFrame(datos)
df = df.fillna(0)
df[['edad','año','fsexual','nhijos','npartos','naborto']].astype('int')
st.sidebar.markdown("### Número de abortos por año ")
select = st.sidebar.selectbox('que quieres visualizar ', ['edad', 'nhijos','nivel_edu'], key='1')
recuento = datos['año'].value_counts()
recuento = pd.DataFrame({'año':recuento.index, 'edad':recuento.values,'nhijos':recuento.values, 'nivel_edu':recuento.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Visualización de los datos")
    if select == 'edad':
        fig = px.bar(recuento, x='año', y='edad', color='edad', height=500)
        st.plotly_chart(fig)
    elif select == 'nhijos':
        fig = px.bar(recuento, x='año', y='nhijos', color= 'año', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.bar(recuento, x='año',y='nivel_edu', color= 'año', height=500)
        st.plotly_chart(fig)

st.markdown("### Motivos para interrumpir el embarazo ")
st.sidebar.subheader("Motivos")
choice = st.sidebar.multiselect('Escoge', ('INTERRUPCION VOLUNTARIA','PROYECTO DE VIDA','SITUACION ECONOMICA','PROBLEMAS DE SALUD','VIOLACION','FALLA DEL METODO'), key=0)
if len(choice) > 0:
    choice_data = datos[datos.motiles.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='edad', y='motiles',
                         histfunc='count', color='motiles',
                         facet_col='motiles', labels={'MOTIVO':'    '},
                          height=600, width=800)
    st.plotly_chart(fig_0)

st.markdown("### Procedimiento Utilizado para interruptir el embarazo")
st.sidebar.subheader("Procedimiento")
choice = st.sidebar.multiselect('Escoge', ('MEDICAMENTO','ASPIRACIÓN ENDOUTERINA (MANUAL O ELÉCTRICA)','LEGRADO'), key=0)
if len(choice) > 0:
    choice_data = datos[datos.procile_simplificado.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='edad', y='procile_simplificado',
                         histfunc='count', color='procile_simplificado',
                         facet_col='procile_simplificado', labels={'PROCEDIMIENTO':''},
                          height=600, width=800)
    st.plotly_chart(fig_0)

st.markdown("### Método Anticonceptivo Utilizado ")
st.sidebar.subheader("Método")
choice = st.sidebar.multiselect('Escoge', ('NINGUNO','DIU','CONDON','IMPLANTE SUBDERMICO'), key=0)
if len(choice) > 0:
    choice_data = datos[datos.panticoncep.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='edad', y='panticoncep',
                         histfunc='count', color='panticoncep',
                         facet_col='panticoncep', labels={'METODO':''},
                          height=600, width=800)
    st.plotly_chart(fig_0)


st.markdown("### Resultados de aplicar K-means y Clustering")
image = Image.open('Figure_1.png')
st.image(image, caption='Resultados de graficar la edad con los demás atributoss', use_column_width=True)

image2 = Image.open('Figure_2.png')
st.image(image2, caption='', use_column_width=True)

image3 = Image.open('Figure_3.png')
st.image(image3, caption='Elbow Curve después de aplicar k means ', use_column_width=True)

image4 = Image.open('Figure_4.png')
st.image(image4, caption='Predicción de Clusters', use_column_width=True)

image5 = Image.open('Figure_5.png')
st.image(image5, caption='Clusters', use_column_width=True)

image6 = Image.open('Figure_6.png')
st.image(image6, caption='', use_column_width=True)

image7 = Image.open('Figure_7.png')
st.image(image7, caption='', use_column_width=True)
