import streamlit as st
import numpy as np
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import folium
import pandas as pd
import geopandas
import plotly.express as px
from datetime import datetime

pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(layout='wide')
st.title('Data Overview')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geo(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_feature(data):
    #add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data(data):
    f_attributes = st.sidebar.multiselect(' Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect(' Enter zipcode', data['zipcode'].unique())

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
        
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    
    st.dataframe(data)

    c1, c2 = st.beta_columns((1,1))

    #Averge metrics

    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    #merge

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPOCODE', 'TOTAL_HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']
    c1.header('Averge Values')
    c1.dataframe(df, height=500)

    #static Descripitive
    num_att = data.select_dtypes(include=['int64', 'float64'])

    min_ = pd.DataFrame(num_att.apply(np.min))
    max_ = pd.DataFrame(num_att.apply(np.max))

    media = pd.DataFrame(num_att.apply(np.mean))
    mediana = pd.DataFrame(num_att.apply(np.median))
    std = pd.DataFrame(num_att.apply(np.std))

    df_num = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df_num.columns = ['Atributos', 'Max', 'Min', 'Media', 'Mediana', 'STD']
    c2.header('Descripitive Analysis')
    c2.dataframe(df_num, height=500)

    return None

def portifolio_density(data, geofile):
    st.title('Region Overview')
    c1, c2 = st.beta_columns((1,1))

    c1.header('Portifolio Density')
    
    df = data

    # Base Map Folium
    density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()],
                default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
            popup=f'Sold R${row["price"]} on {row["date"]}. Features: {row["sqft_living"]} sqft, {row["bedrooms"]} bedrooms, {row["bathrooms"]} bathrooms, year built {row["yr_built"]}').add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Regiom Price map
    c2.header('Price Density')

    
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns =['ZIP', 'PRICE']


    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                default_zoom_start=15)


    region_price_map.choropleth(data=df,
                                geo_data = geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')


    with c2:
        folium_static(region_price_map)

    return None

def commercial_distribution(data):
    st.sidebar.title('Commercial options')
    st.title('Commercial Attributes')

    c1, c2 = st.beta_columns((1,1))

    # ----------------- Averge Price per Year

    # Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    # Data selectio
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()


    # Visualização Gráfica
    c1.header('Averge Price per Year')
    fig = px.line(df, x="yr_built", y="price")
    c1.plotly_chart(fig, use_container_width=True)

    # ----------------- Averge Price per Day

    # Filters
    min_date = datetime.strptime("2014-05-02", "%Y-%m-%d")
    max_date = datetime.strptime("2015-05-27", "%Y-%m-%d")

    st.sidebar.subheader('Select Max Date')
    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    #data selectio
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # Visualização Gráfica
    c2.header('Averge Price per Day')
    fig = px.line(df, 'date', 'price')
    c2.plotly_chart(fig, use_container_width=True)

    # ----------------- Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')
    lista = data['price'].unique().tolist()

    # Filters
    avg_price = data['price'].mean()
    f_price = st.sidebar.select_slider('Price', options=sorted(lista))

    # Visualização Gráfica
    df = data[data['price'] <= f_price]
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    c1, c2 = st.beta_columns(2)

    # ----------------- Histograma
    c1.header('Bedrooms Distribution')
    st.sidebar.subheader('Select Max Bedrooms')

    # Filters
    f_bedrooms = st.sidebar.selectbox('Bedrooms', sorted(data['bedrooms'].unique()))

    # Visualização Gráfica
    df = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms')
    c1.plotly_chart(fig, use_container_width=True)

    # ----------------- Histograma
    c2.header('Bathrooms Distribution')
    st.sidebar.subheader('Select Max Bathrooms')

    # Filters
    f_bathrooms = st.sidebar.selectbox('Bathrooms', sorted(data['bathrooms'].unique()))

    # Visualização Gráfica
    df = data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram(df, x='bathrooms')
    c2.plotly_chart(fig, use_container_width=True)

    # ----------------- Histograma
    c1, c2 = st.beta_columns(2)

    c1.header('Floors Distribution')
    st.sidebar.subheader('Select Max Floors')

    # Filters
    f_floors = st.sidebar.selectbox('Floors', sorted(data['floors'].unique()))

    # Visualização Gráfica
    df = data[data['floors'] <= f_floors]
    fig = px.histogram(df, x='floors')
    c1.plotly_chart(fig, use_container_width=True)

    # ----------------- Histograma
    c2.header('Waterfront Distribution')
    st.sidebar.subheader('Select Water View')

    # Filters

    f_waterfront = st.sidebar.checkbox('waterfront', value=False)

    # Visualização Gráfica
    df = data[data['waterfront'] == f_waterfront]
    fig = px.histogram(df, x='waterfront')
    c2.plotly_chart(fig, use_container_width=True)



if __name__ == '__main__':
    #ETL
    #Data extraction
    path = r'C:\Users\douglas\Desktop\DG\house_rocket\kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    
    data = get_data(path)
    
    geofile = get_geo(url)
    #Transformation
    data['date'] = pd.to_datetime(data['date'])
    data = set_feature(data)

    overview_data(data)

    portifolio_density(data, geofile)

    commercial_distribution(data)
    
    attributes_distribution(data)

