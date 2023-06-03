#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


# In[4]:


st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://docs.streamlit.io/streamlit-cloud/troubleshooting',
         'Report a bug': "https://github.com/streamlit/streamlit/issues",
         'About': "Car Price EDA"
     }
 )


# In[8]:


# loading dataset
def load_data():
    df = pd.read_csv('cpp.csv')
    df = df.drop('ID', axis =1 )
    df.rename(columns = {'Drive wheels':'Drive_Wheels','Gear box type':'gear_box', 'Fuel type':'FuelType','Prod. year':'Year','Leather interior': 'Interior','Engine volume': 'Volume' }, inplace = True)
    df=df.drop_duplicates(keep='first')
    df['Levy'] =df['Levy'].replace(['-'], 0)
    df['Levy']=df['Levy'].astype(str).astype(int)
    df['Mileage']=df['Mileage'].apply(lambda x:str(x).replace("km"," "))
    df['Mileage']=df['Mileage'].astype(str).astype(int)
    df['Volume']=df['Volume'].apply(lambda x:str(x).replace("Turbo"," "))
    df['Volume']=df['Volume'].astype(str).astype(float)
    return df

df = load_data()


# In[10]:


df.info()



#Start building Streamlit App
st.title('Car Price Prediction')
st.text("")


st.sidebar.image('https://repository-images.githubusercontent.com/286819592/b82e14cf-3c85-4f91-84c0-bea095c353a8', width=250)

menu = st.sidebar.radio(
    "Menu:",
    ('Introduction', "Analysis", "Model"),)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project By: Trixie Ann Galang')
st.sidebar.write('[Github](https://github.com/TrixieAnnGalang/Car-price-prediton.git')

if menu == 'Introduction':
    st.header('Data Overview')
    st.write('A dataset to practice regression by predicting the prices of different cars.')
    st.write('CSV file - 19237 rows x 18 columns (Includes Price Columns as Target')
    st.write(' Attributes: ID,Price: price of the care(Target Column)Levy,Manufacturer,Model,Prod. year,Category,Leather interior,Fuel type,Engine volume,Mileage,Cylinders,Gear box type,Drive wheels,Doors,Wheel,Color,Airbags')
    st.header('Data')
    st.dataframe(df.describe())
    st.write('About data')
    col1, col2 = st.columns(2)
    with col1:
        st.write('First 5 Rows')
        st.dataframe(df.head())
    with col2:
        st.write('Last 5 Rows')
        st.dataframe(df.tail())


elif menu == 'Analysis':
    st.write('Lets Do EDA')
    st.header('Cars Produced')
    cat = st.selectbox('Number of cars produced by:',('Manufacturer','Catagory','Model'),) 
    if cat == 'Manufacturer':
        fig1 = plt.figure(figsize=(15,10))
        Manufacturer = df.Manufacturer.value_counts()
        plt.title('Different Manifacturing companies')
        sns.set_style('dark')
        m = sns.barplot(x=Manufacturer.index, y = Manufacturer, palette ='Paired')
        m.set_xticklabels(Manufacturer.index, rotation = 90)
        m.set(xlabel ='Manufacturer', ylabel = 'Values')
        st.pyplot(fig1)
    elif cat=='Catagory':
        Model = df.Model.value_counts().head(40)
        fig2=plt.figure(figsize=(15,10))
        plt.title('Different Model Cars', fontsize =12)
        mode = sns.barplot(Model.index, Model, palette ='Paired')
        sns.set_style('dark')
        mode.set_xticklabels(Model.index, rotation = 90)
        mode.set(xlabel= 'Car Model', ylabel = 'Values')
        st.pyplot(fig2)
        
    elif cat == 'Model':
        Category = df.Category.value_counts()
        fig3=plt.figure(figsize=(10,6))
        plt.tight_layout(pad=2)
        font = {'family' : 'monospace',
                'weight' : 'bold',
                }
        plt.rc('font', **font)
        plt.title( 'Cars in each category')
        sns.set_style("white")
        category_car=sns.barplot(Category.index,Category);
        category_car.set_xticklabels(Category.index ,rotation=90)
        category_car.set(xlabel='Category Name', ylabel='Number of Cars')
        st.pyplot(fig3)
        
    st.header('Tree Map')
    tree= st.selectbox(':',('Cars','Price'),)
    if tree == 'Cars':
        distribution = px.treemap(data_frame =df, path= ['Manufacturer', 'Category','Model'])
        st.plotly_chart(distribution)
    elif tree == 'Price':
        distribution = px.treemap(data_frame=df,path=["Manufacturer","Category","Model"],values='Price')

    col3,col4= st.columns(2)
    with col3:
        st.header('Mean Price')
        feature= st.selectbox('Select features:',('Manufacturer','gear_box','FuelType','Category','Interior','Doors','Cylinders','Volume','Color',),)
        bar = pd.DataFrame(df.groupby(feature)['Price'].mean().reset_index()).sort_values(by = feature)
        fig4 = px.bar(bar, x=feature, y= 'Price',)
        st.plotly_chart(fig4)
    with col4:
        st.header('Car Distribution')
        features= st.selectbox('Select features:',('gear_box','FuelType','wheel','Drive_Wheels'),)
        pie_chart = df[features]
        fig5 = px.pie(pie_chart ,names = features)
        st.plotly_chart(fig5)


    st.header('Levy')
    fig6=plt.figure(figsize=(25,15))
    year = sns.lineplot(x = 'Year', y = 'Price', data=df)
    plt.title('Prodcution year')
    st.pyplot(fig6)
        
elif menu == 'Model':
    st.header('Model Performance')
    st.write('I have applied three different regression models; the Liner regression, Lasso regression and Random forest. First, I selected all the features for the models then selected the the top ten features using the Chi squired method for the models. For model evaluation I used the Mean absolute error. ' )
    method= st.selectbox('Select method:',('All features','top 10 features'),)
    if method == 'All features':
        st.subheader('Linear Regression Model')
        st.write('*Mean absolute error =13183.447333882756*')
        st.subheader('Lasso Regression Model')
        st.write('*Mean absolute error = 13183.083024324877*')
        st.subheader('Random Forest')
        st.write('*Mean absolute error = 11828.475715567669*')
    else:
        st.subheader('Feature selection using chi2')
        st.write('k = 10: Manufacturer, Interior, FuelType, Mileage, gear_box, Drive_Wheels, Doors,Wheel, Color, Airbags')
        st.subheader('Linear Regression Model')
        st.write('*Mean Absolute Error = 14037.961427893164*')
        st.subheader('Lasso regression')
        st.write('*Mean absolute error = 14037.696924963304*')
        st.subheader('Random Forest')
        st.write('*Mean Absolute error =9925.74945198783*')



