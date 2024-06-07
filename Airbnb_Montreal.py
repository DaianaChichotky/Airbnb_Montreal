# --------------------LIBRERÍAS-----------------------------------------------#

import streamlit as st
from streamlit_option_menu import option_menu
import base64

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import os
import json

# interactive maps
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd
from branca.colormap import LinearColormap
from folium.plugins import HeatMap
from folium.features import GeoJsonTooltip
import streamlit.components.v1 as components

# Plotly graphs
import plotly.graph_objs as go
import plotly_express as px

# AB testing
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import mannwhitneyu

# Prediction
import pickle
from pycaret.regression import load_model, predict_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)


# --------------------SITE CONFIGURATION-------------------------------------#

st.set_page_config(page_title='Airbnb Montreal',
                   layout='wide',
                   page_icon='🏠')


# --------------------COLUMNS-----------------------------------------------#

# Center the image
col1, col2, col3 = st.columns([3, 1, 3])

with col2:
    
    st.image("img/Airbnb_icon-removebg-preview.png", width=200)
    
st.markdown("<h1 style='text-align: center; margin-top: -40px; color:#FD676C;'>Análisis de alquileres en Montreal</h1>" ,unsafe_allow_html=True)
    

# ---------------------BACKGROUND IMAGE---------------------------------------#

def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
            .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
add_bg_from_local(r"img/background2.png")   

    
# ---------------------MENU-----------------------------------------------------# 

# Menu
page = option_menu(
    menu_title = None,
    options = ["Introducción", "Análisis de barrios", "Análisis de alojamientos", "Análisis de disponibilidad", "Análisis de precios",
               "PowerBi Reseñas y Superhost", "Galería", "Predicción de precios", "Conclusiones"], 
    icons=["info-circle", "pin", "house", "calendar-check", "coin", "bar-chart","camera","arrow-up-right", "check-circle"],
    default_index=0,
    orientation="horizontal",
      styles={
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "padding": "0px", "--hover-color": "#eee"},
        "icon": {"margin": "auto", "display": "block"}})


# --------------------DATA LOADING-----------------------------------------------#

listings1 = pd.read_csv("notebooks/listings1_cleaned.csv")
#reviews = pd.read_csv("notebooks/reviews_cleaned.csv")
#calendar = pd.read_csv("notebooks/calendar_cleaned.csv")
neighbourhoods_geojson = gpd.read_file("https://data.insideairbnb.com/canada/qc/montreal/2024-03-23/visualisations/neighbourhoods.geojson")


####################################  PAGE 1  ######################################

if page =="Introducción":

    st.markdown("<h3 style='text-align: left; margin-top: -20px; color:#FD676C;'>Introducción</h3>" ,unsafe_allow_html=True)

    st.markdown("""***Somos una consultora especializada en el análisis del mercado de alquileres en Airbnb en Montreal. Nuestro objetivo es proporcionar recomendaciones estratégicas para maximizar los ingresos y las reseñas positivas de nuestros clientes.***""")
    st.markdown("***El objetivo de esta plataforma es identificar las áreas clave de oportunidad para nuestros clientes a través de la exploración de datos recopilados de la plataforma InsideAirbnb.***")
    st.markdown("***Al final del análisis, nuestros clientes obtendrán recomendaciones claras y accionables que los ayuden a tomar decisiones informadas y estratégicas.***")

    st.markdown("<h3 style='text-align: left; color: #FD676C;'>Dataset</h3>" ,unsafe_allow_html=True)

    # I create a copy of the dataset listings1 to work with it and to keep also the original dataset.
    df = listings1.copy()

    df = df[['name', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost',
       'neighbourhood', 'room_type',
       'property_type', 'bathrooms', 'bedrooms',
       'beds', 'accommodates', 'price', 'review_scores_rating',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'has_availability',
       'minimum_nights', 'maximum_nights']]

    # I change the name of the columns
    nombres_columnas = {
     'name' : 'Nombre',
     'host_response_rate':'Host response rate',
     'host_acceptance_rate':'Host acceptance rate',
     'host_is_superhost':'Superhost',
       'neighbourhood':'Barrio',
       'room_type': 'Tipo de alojamiento',
       'property_type': 'Tipo de propiedad',
       'bathrooms':'Nº de baños',
       'bedrooms':'Nº de dormitorios',
       'beds':'Nº de camas',
       'accommodates':'Nº de huéspedes',
       'price':'Precio por noche',
       'review_scores_rating':'Review scores rating',
       'review_scores_cleanliness':'Review scores cleanliness',
       'review_scores_checkin':'Review scores checkin',
       'review_scores_communication':'Review scores communication',
       'review_scores_location':'Review scores location',
       'has_availability':'Disponibilidad',
       'minimum_nights':'Mínimo de noches',
       'maximum_nights':'Máximo de noches'}

    df = df.rename(columns=nombres_columnas)

    # I change the order of the columns
    nuevo_orden_columnas = ['Nombre', 'Barrio', 'Tipo de alojamiento', 'Tipo de propiedad',
                        'Nº de baños', 'Nº de dormitorios', 'Nº de camas',
                        'Nº de huéspedes', 'Disponibilidad', 'Mínimo de noches', 'Máximo de noches',
                        'Precio por noche', 'Superhost',
                        'Host response rate', 'Host acceptance rate',
                        'Review scores rating', 'Review scores cleanliness',
                        'Review scores checkin', 'Review scores communication',
                        'Review scores location']

    df = df[nuevo_orden_columnas]

    # I change the description of the values
    df['Disponibilidad'] = df['Disponibilidad'].replace({'t': 'Disponible', 'f': 'No disponible'})


    # --------------------SIDEBAR-------------------------------------#

    st.sidebar.image('img/background1.png', use_column_width=True)
    st.sidebar.title("Filtros")
    st.sidebar.write('-------')

    # ------------------ FILTERS -------------------------------------#

    with st.sidebar:
          # Filter neighbourhood
          filter_barrio = st.multiselect("Barrio", df["Barrio"].unique())
          
          # Filter room type
          filter_room = st.multiselect("Tipo de alojamiento", df["Tipo de alojamiento"].unique())
    
          # Filter accommodates
          filter_acc = st.slider('Nº de huéspedes', min_value=1, max_value=16, value=(0, 16))
    
          # Filter price
          filter_price = st.slider("Precio por noche", min_value=0, max_value=300, value=(0, 300))
    
          # Filter minimum nights
          filter_noches = st.slider("Mínimo de noches", min_value=1, max_value=365, value=(1, 365))
     
          # Aplicar filtros al DataFrame
          df_filtrado = df.copy()


    if filter_barrio:
          df_filtrado = df_filtrado[df_filtrado['Barrio'].isin(filter_barrio)]
    if filter_room:
          df_filtrado = df_filtrado[df_filtrado['Tipo de alojamiento'].isin(filter_room)]

    df_filtrado = df_filtrado[(df_filtrado['Nº de huéspedes'] >= filter_acc[0]) & (df_filtrado['Nº de huéspedes'] <= filter_acc[1])]

    df_filtrado = df_filtrado[(df_filtrado['Precio por noche'] >= filter_price[0]) & (df_filtrado['Precio por noche'] <= filter_price[1])]

    df_filtrado = df_filtrado[(df_filtrado['Mínimo de noches'] >= filter_noches[0]) & (df_filtrado['Mínimo de noches'] <= filter_noches[1])]

    
    # Dataframe :

    st.dataframe(df_filtrado)

    # Resultados obtenidos del filtrado
    resultado_df = df_filtrado.shape[0]
    st.write(f"<div style='color: #ff5a60; text-align:center; font-size: 24px;'>Resultados obtenidos: <b>{resultado_df}</b></div>", unsafe_allow_html=True)
    st.markdown('---------')
    st.markdown('***El dataset completo se encuentra en https://insideairbnb.com/get-the-data/***')
    

####################################  PAGE 2  ##########################################

if page == "Análisis de barrios":

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        
        tab1, tab2 = st.tabs([
        "Mapa de barrios",
        "Mapa de precios"])
    
    # ----------------------- tab 1  --------------------------------#
    
    with tab1:
        
        # Mapa de barrios
        
        st.markdown("<h5 style='text-align: left;'>¿Cómo es la distribución de los alojamientos de Airbnb en Montreal?</h5>", unsafe_allow_html=True)
        st.markdown('**Haz zoom en el mapa:**')

        with open('./HTML/map1_barrios.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=500)

        st.markdown('**El barrio donde se encuentran la mayoría de los alojamientos es Ville-Marie, en el centro de la ciudad.**')

        # Distribución de alojamientos

        with open('./HTML/distribucion_alojamientos.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=1000)

    # ----------------------- tab 2  ----------------------------------#

    with tab2:

        # Mapa de precios
        
        st.markdown("<h5 style='text-align: left;'>¿Cómo es la distribución de los precios promedio por noche en Montreal?</h5>", unsafe_allow_html=True)
        st.markdown('**Haz zoom en el mapa:**')
            
        with open('./HTML/map2_prices.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=500)

        st.markdown("<h5 style='text-align: left; '>A través del test estadístico de Skewness se comprueba que el precio del alojamiento no depende de su ubicación.</h5>", unsafe_allow_html=True)
        
        # Precio por noche por barrio  
        
        st.markdown('**El barrio con el precio promedio por noche más elevado es *Saint-Geneviève*:**')
        
        with open('./HTML/precio_barrio.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=1000)
       
  
####################################  PAGE 3  ##########################################

if page == "Análisis de alojamientos":  


    col1, col2 = st.columns(2)

    with col1:

        # Cantidad de ofertas por tipo de alojamiento
                
        with open('./HTML/room_type.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=500)

        st.markdown("<h4 style='text-align: center; margin-top: -10px;'>82% Entire Home/Apartments.</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; margin-top: -10px;'>17% Habitaciones privadas.</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; margin-top: -10px;'>0.5% Habitaciones compartidas y hotel.</h4>", unsafe_allow_html=True)

    with col2:

        # Cantidad de ofertas por número de huéspedes
         
        with open('./HTML/accommodates.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=500)
     
       # st.markdown("<h4 style='text-align: center; margin-top: -10px;'>El 37.5 % de los alojamientos es para 2 huéspedes.</h4>", unsafe_allow_html=True)
        
####################################  PAGE 4  ##########################################

if page == "Análisis de disponibilidad":
     
    col1, col2, col3 = st.columns(3)
    
    with col1:

        # Mapa de calor de precios
        
        st.markdown("<h5 style='text-align: center;'>Mapa de calor con las zonas de mayor disponibilidad:</h5>", unsafe_allow_html=True)

        with open('./HTML/map4_availability.html', 'r', encoding='utf-8') as f:
             html_content = f.read()
             st.components.v1.html(html_content, height=700)

    with col2:

        # Disponibilidad por barrio

        st.write("")
        st.write("")
        
        with open('./HTML/availability_neigh.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=700)
        

    with col3:

        # Disponibilidad por tipo de alojamiento

        st.write("")
        st.write("")
        
        with open('./HTML/availability_roomtype.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=700)


####################################  PAGE 5  ##########################################

if page == "Análisis de precios":  

 #   centered_tabs_css = """
 #   <style>
 #   div.stTabs {
 #       display: flex;
 #       justify-content: center;
 #   }
 #   </style>
 #  """

   # st.markdown(centered_tabs_css, unsafe_allow_html=True)
     
  tab1, tab2, tab3 = st.tabs([
  "Precio VS tipo de alojamiento",
  "Precio VS Huéspedes",
  "Evolución de precios"
  ])
  
  with tab1:
    
    # Precio VS tipo de alojamiento
    
    with open('./HTML/precio_roomtype.html', 'r', encoding='utf-8') as f:
      html_content = f.read()
      st.components.v1.html(html_content, height = 500)
      st.markdown("<h4 style='text-align: center; margin-top: -50px;'>A través del método de Skewness puedo confirmar que el tipo de propiedad tiene un impacto significativo en el precio de la vivienda.</h4>", unsafe_allow_html=True)

  with tab2:
    # Precio VS Huéspedes
    
    with open('./HTML/precio_accommodates.html', 'r', encoding='utf-8') as f:
      html_content = f.read()
      st.components.v1.html(html_content, height=500)

  with tab3:
    # Evolución de precios
    
    with open('./HTML/grafico_price_evolution.html', 'r', encoding='utf-8') as f:
      html_content = f.read()
      st.components.v1.html(html_content, height=500)
      st.markdown("<h4 style='text-align: center; margin-top: -50px;'> Los precios pueden variar según temporada y eventos locales.</h4>", unsafe_allow_html=True)


####################################  PAGE 6  ##########################################


if page == "PowerBi Reseñas y Superhost": 
    
    # Enlace del Power BI

    powerbi_embed_url = "https://app.powerbi.com/view?r=eyJrIjoiMzRhMzRiYzUtY2E5OC00Yzg2LTlkYWEtYTg0ZTg0MDZmYmExIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9"

    iframe_code = f"""<div style="display: flex; justify-content: center; margin-top: 20px;"><iframe width="1140" height="541.25" src="{powerbi_embed_url}" frameborder="0" allowFullScreen="true"></iframe></div>"""

    st.markdown(iframe_code, unsafe_allow_html=True)

    #----------------------------------#
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
            
            # Correlación 

            st.markdown('')
            st.markdown("<h3 style='text-align: center; '>Correlación de variables</h3>" ,unsafe_allow_html=True)
            
            # Cambiar nombres de las variables
            
            variables = listings1[['host_response_rate', 'host_acceptance_rate',
                       'host_listings_count',
                       'accommodates', 'price', 'number_of_reviews', 'review_scores_rating',
                       'review_scores_cleanliness', 'review_scores_checkin',
                       'review_scores_communication', 'review_scores_location',
                       'reviews_per_month']].rename(columns={
    'host_response_rate': 'Respuesta del anfitrión',
    'host_acceptance_rate': 'Puntuación del anfitrión',
    'host_listings_count': 'Nº de listados por anfitrión',
    'accommodates': 'Nº de huéspedes',
    'price': 'Precio',
    'number_of_reviews': 'Nº de reseñas',
    'review_scores_rating': 'Puntuación general',
    'review_scores_cleanliness': 'Limpieza',
    'review_scores_checkin': 'Check-in',
    'review_scores_communication': 'Comunicación',
    'review_scores_location': 'Ubicación',
    'reviews_per_month': 'Nº de reseñas por mes'})
            
            # Mapa de calor para analizar la relacion entre las variables continuas. Use el método "spearman" ya que las variables no siguen una distribución normal.

            corr = variables.corr(method='spearman').sort_values(by='Puntuación general', axis=0, ascending=False).sort_values(by='Puntuación general', axis=1, ascending=False)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            f, ax = plt.subplots(figsize=(12, 12))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                        square=True, linewidths=0, cbar_kws={"shrink": .5}, annot=True)  # annot=True para ver los valores en el gráfico
            
            plt.tight_layout()

            st.pyplot(plt)

   
    #------------ Col 2 -------------#

    with col2:
        
        st.markdown('')

        st.markdown("<h3 style='text-align: center; '>Análisis de la correlación:</h3>" ,unsafe_allow_html=True)
        
        st.markdown("""<h5 style='text-align: left; '>Número positivo (0 a 1):\n
                    Indica una correlación positiva""" ,unsafe_allow_html=True)
        
        st.markdown("""<h5 style='text-align: left; '>Número negativo(-1 a 0):\n
                    Indica una correlación negativa""" ,unsafe_allow_html=True)
        
        st.markdown("""<h5 style='text-align: left; '>Cero (0):\n 
                    Indica que no hay correlación entre las variables.""" ,unsafe_allow_html=True)
        
        st.markdown("<h5 style='text-align: left; color:#FD676C;'><b>La puntuación tiene una fuerte correlación positiva con la limpieza, comunicación, check-in y ubicación del alojamiento.</b></h5>" ,unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: left; color:#FD676C;'><b>El precio tiene una fuerte correlación positiva con el número de huéspedes.</b></h5>" ,unsafe_allow_html=True)

####################################  PAGE 7  ##########################################


if page == "Galería":
     
    # Sidebar:
     
    st.sidebar.image('./img/sidebar2.jpg', use_column_width=True)
    st.sidebar.title("Filtros")
    st.sidebar.write('-------')

    with st.sidebar:
          
        # Filter neighbourhood
        filter_barrio = st.multiselect("Barrio", listings1["neighbourhood"].unique())
          
        # Filter room type
        filter_room = st.multiselect("Tipo de alojamiento", listings1["room_type"].unique())
    
        # Filter accommodates
        filter_acc = st.slider('Número de huéspedes', min_value=1, max_value=16, value=(0, 16))
    
        # Filter price
        filter_price = st.slider("Precio por noche", min_value=0, max_value=300, value=(0, 300))
    
        # Filter minimum nights
        filter_noches = st.slider("Mínimo de noches", min_value=1, max_value=365, value=(1, 365))
     
        # Aplicar filtros al DataFrame
        df_filtrado = listings1.copy()


    if filter_barrio:
        df_filtrado = df_filtrado[df_filtrado['neighbourhood'].isin(filter_barrio)]
    if filter_room:
        df_filtrado = df_filtrado[df_filtrado['room_type'].isin(filter_room)]

    df_filtrado = df_filtrado[(df_filtrado['accommodates'] >= filter_acc[0]) & (df_filtrado['accommodates'] <= filter_acc[1])]

    df_filtrado = df_filtrado[(df_filtrado['price'] >= filter_price[0]) & (df_filtrado['price'] <= filter_price[1])]

    df_filtrado = df_filtrado[(df_filtrado['minimum_nights'] >= filter_noches[0]) & (df_filtrado['minimum_nights'] <= filter_noches[1])]

    
    # I create columns

    col1, col2 = st.columns(2)

    # ------- Column 1 ------#
    
    with col1:
        
        st.markdown("<h5 style='text-align: center; margin-top: -20px; color:#FD676C;'>Imágenes de algunos alojamientos seleccionados</h5>" ,unsafe_allow_html=True)
        
        # Mostrar imágenes filtradas
        
        if 'picture_url' in df_filtrado.columns and not df_filtrado.empty:
            num_images = min(len(df_filtrado), 16)
            images_per_row = 2
            rows = (num_images + images_per_row - 1) // images_per_row
            
            for i in range(rows):
                cols = st.columns(images_per_row)
                for j in range(images_per_row):
                    index = i * images_per_row + j
                    if index < num_images:
                        with cols[j]:
                            try:
                                st.image(df_filtrado.iloc[index]['picture_url'],
                                         caption=df_filtrado.iloc[index].get('name', f"Imagen {index+1}"),
                                         width=200)  # Ancho de la imagen ajustado a 200 píxeles
                            except Exception as e:
                                st.write(f"Error al cargar la imagen en el índice {index}: {e}")
                                
            else:
                st.write("")

    # ------- Column 2 ------#

    with col2:

        # Elijo las columnas que me interesan del dataframe:

        df_filtrado = df_filtrado[['name', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                                   'property_type', 'bathrooms', 'bedrooms','beds', 'price', 'review_scores_rating',
                                   'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication', 'review_scores_location',
                                   'has_availability']]

        # I change the name of the columns
        nombres_columnas = { 
             'name' : 'Nombre',
             'host_response_rate':'Host response rate',
             'host_acceptance_rate':'Host acceptance rate',
             'host_is_superhost':'Superhost',
             'property_type': 'Tipo de propiedad',
             'bathrooms':'Nº de baños',
             'bedrooms':'Nº de dormitorios',
             'beds':'Nº de camas',
             'price':'Precio por noche',
             'review_scores_rating':'Review scores rating',
             'review_scores_cleanliness':'Review scores cleanliness',
             'review_scores_checkin':'Review scores checkin',
             'review_scores_communication':'Review scores communication',
             'review_scores_location':'Review scores location',
             'has_availability':'Disponibilidad'
             }

        df_filtrado = df_filtrado.rename(columns=nombres_columnas)

        # I change the order of the columns
        nuevo_orden_columnas = ['Nombre', 'Tipo de propiedad', 'Disponibilidad',
                        'Nº de dormitorios', 'Nº de camas', 'Nº de baños',
                        'Precio por noche', 'Superhost',
                        'Host response rate', 'Host acceptance rate',
                        'Review scores rating', 'Review scores cleanliness',
                        'Review scores checkin', 'Review scores communication',
                        'Review scores location']

        df_filtrado = df_filtrado[nuevo_orden_columnas]

        # I change the description of the values
        df_filtrado['Disponibilidad'] =  df_filtrado['Disponibilidad'].replace({'t': 'Disponible', 'f': 'No disponible'})

        st.markdown("<h5 style='text-align: center; margin-top: -20px; color:#FD676C;'>Información sobre los alojamientos seleccionados</h5>" ,unsafe_allow_html=True)

        st.dataframe(df_filtrado)
        # Resultados obtenidos del filtrado
        resultado_df_filtrado = df_filtrado.shape[0]
        
        st.write(f"<div style='color: #ff5a60; text-align:center;'>Resultados obtenidos: <b>{resultado_df_filtrado}</b></div>", unsafe_allow_html=True)
         
        # Resultados obtenidos del filtrado
        resultado_df_filtrado = df_filtrado.shape[0]
        

####################################  PAGE 8  ##########################################


if page == "Predicción de precios":

    col1, col2, col3 =st.columns(3)

    with col2:
        
        st.markdown("<h4 style='text-align: center; color:#FD676C;'><b>Predicción de precios para alojamientos en Montreal</b></h4>", unsafe_allow_html=True)

        st.markdown("<h6 style='text-align: left;'><b>Selecciona los filtros:</b></h6>", unsafe_allow_html=True)

        # Cargo el modelo 
        model = load_model("notebooks/airbnb_predictor")

        # Campos de entrada
        neighbourhood = st.selectbox('Barrio',
                                 ['Ville-Marie', 'Le Plateau-Mont-Royal',
                                  'Rosemont-La Petite-Patrie', 'Côte-des-Neiges-Notre-Dame-de-Grâce',
                                  'Le Sud-Ouest', 'Villeray-Saint-Michel-Parc-Extension',
                                  "Baie-d'Urfé", 'Saint-Laurent', 'Mercier-Hochelaga-Maisonneuve',
                                  'Verdun', 'Lachine', 'Outremont', 'Ahuntsic-Cartierville',
                                  'Westmount', 'Rivière-des-Prairies-Pointe-aux-Trembles', 'Anjou',
                                  'Pointe-Claire', 'Mont-Royal', 'LaSalle', 'Côte-Saint-Luc',
                                  'Hampstead', "L'Île-Bizard-Sainte-Geneviève",
                                  'Pierrefonds-Roxboro', 'Saint-Léonard', 'Dorval',
                                  'Dollard-des-Ormeaux', 'Montréal-Nord', 'Beaconsfield',
                                  'Montréal-Ouest', 'Kirkland', 'Montréal-Est',
                                  'Sainte-Anne-de-Bellevue', "L'Île-Dorval"])
    
        accommodates = st.number_input('Número de huéspedes',
                                   min_value=1, max_value=16)
       
        room_type = st.selectbox('Tipo de propiedad',
                             ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
    
        bedrooms = st.number_input('Número de camas',
                 min_value=1, max_value=16)
    
        bathrooms = st.number_input('Número de baños',
                 min_value=1, max_value=16)

        minimum_nights = st.number_input('Mínimo de noches', 
                                     min_value=1, max_value=365)

        # Creo un df con los inputs 
        input_data = pd.DataFrame({
         'neighbourhood': neighbourhood,
         'accommodates': accommodates,
         'room_type': room_type,
         'bedrooms': bedrooms,
         'bathrooms': bathrooms,
         'minimum_nights': minimum_nights},
         index=[0])

        # Hacer la prediccion

        st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FD676C;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 20px;
        margin: auto;
        display: block;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)
    
        # Creo una columna para centrar el botón dentro de ella
        col1, col2, col3 = st.columns([1,3,1])
        
        with col2:
            
            if st.button('Predecir precio'):
                prediction = predict_model(model, data=input_data)
                prediccion = prediction['prediction_label'][0].round(2)
                st.markdown(f"<h3 style='text-align: center; margin-top: -20px; '> La predicción del precio por noche para el alojamiento es de {prediccion} $ </h3>", unsafe_allow_html=True)


####################################  PAGE 9  ##########################################


if page == "Conclusiones":

    col1, col2, col3 =st.columns([1,2,1])

    with col2:

        # Título
        
        st.markdown("<h2 style='text-align: center; font-size:35px;'>Conclusiones del análisis de alquileres de Airbnb en Montreal</h2>", unsafe_allow_html=True)

        #st.markdown("<h4 style='text-align: left ;'><b>📌 Ubicación (barrio)</b></h4>", unsafe_allow_html=True)
        st.subheader("📌 Ubicación")
        st.write("- **Alta concentración en el centro**: Le Plateau-Mont-Royal y Ville-Marie son los más populares.")

        st.subheader("🏠 Tipo de Alojamiento")
        st.write("- **82% Entire Home/Apartments**: Más privacidad, preferido por familias.")
        st.write("- **17% Private Rooms**: Económico, ideal para viajeros individuales.")
        st.write("- **0.5% Shared Rooms**: Opción más económica.")
        st.write("- **0.5% Hotel Rooms**: Muy poca oferta, no son populares en Airbnb.")

        st.subheader("📅 Disponibilidad")
        st.write("- **Centro**: Alta disponibilidad y demanda.")
        st.write("- **Barrios periféricos**: Disponibilidad variable, picos durante eventos.")
        st.write("- **Entire Home > Private Rooms > Shared room > Hotel room**: En términos de disponibilidad.")

        st.subheader("💰 Precios")
        st.write("- **Centro**: $130 promedio por noche.")
        st.write("- **Periferia**: $100 promedio por noche.")
        st.write("- **Hotel room > Entire Home > Private Rooms = Shared Rooms**: En términos de costos.")
        st.write("- **A mayor número de huéspedes, más costoso es.**")

        st.subheader("⭐ Reseñas de los huéspedes")
        st.write("- **Limpieza y Atención al Cliente**: Factores clave para altas calificaciones.")
        st.write("- **Ubicación y Comodidad**: Factores clave para reseñas positivas.")

        st.markdown("<h2 style='text-align: center; font-size:35px;'>Recomendaciones Estratégicas</h2>", unsafe_allow_html=True)
       
        st.write("1. **Ajuste de Precios**: Según temporada y eventos locales.")
        st.write("2. **Mejora de Calidad**: Mantener altos estándares de limpieza, comodidad y comunicación.")
        st.write("3. **Exploración de Nuevos Barrios**: Oportunidades en áreas emergentes.")
        st.write("4. **Atención al Cliente**: Respuestas rápidas y efectivas para una mejor experiencia.")





       
       
        
        
        
        
        
        
        
        
        
        
        

    
