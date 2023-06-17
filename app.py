import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


st.title('Cars Price Prediction')
st.image('car.png')
#Load data
df = pd.read_csv('Clean_Data.csv')
preprocessor = pickle.load(open('Preprocessor.pkl','rb'))
model = pickle.load(open('Model.pkl','rb'))

# App

Name = st.selectbox('Car Model',df['Name'].unique())
Location = st.selectbox('Location',df['Location'].unique())
year = st.number_input('Year',df['Year'].min() ,df['Year'].max())
Kilometers_Driven = st.number_input('Kilometers Driven',df['Kilometers_Driven'].min()  ,df['Kilometers_Driven'].max()  )
Fuel_Type = st.selectbox('Fuel Type',df['Fuel_Type'].unique())
Transmission = st.selectbox('Transmission',df['Transmission'].unique())
Owner_Type = st.selectbox('Owner Type',df['Owner_Type'].unique())
Mileage = st.number_input('Mileage',df['Mileage'].min() ,df['Mileage'].max() )
Engine = st.number_input('Engine',df['Engine'].min() ,df['Engine'].max() )
Power = st.number_input('Power',df['Power'].min() ,df['Power'].max() )
Seats = st.selectbox('Seats',df['Seats'].unique())

new_data = {'Name': Name , 'Location': Location ,'Year':year , 'Kilometers_Driven':Kilometers_Driven ,
            'Fuel_Type':Fuel_Type , 'Transmission':Transmission , 'Owner_Type':Owner_Type , 
           'Mileage':Mileage,'Engine':Engine,'Power':Power , 'Seats':Seats}
new_data = pd.DataFrame(new_data,index=[0])

#Preprocessed
new_data_preprocessed = preprocessor.transform(new_data)

log_price = model.predict(new_data_preprocessed)
price = np.expm1(log_price)
price_usd = price[0] * 1219

# Output
if st.button('Predict'):
    st.markdown('## Price in USD:')
    st.markdown(price_usd.round(2))
