import xgboost 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.preprocessing import OrdinalEncoder
loaded_model = pickle.load(open('model.pkl', 'rb'))

@st.cache
def predict_result(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction == 0 :
        return "This restaurant is not recommended 👎🙅‍♂️"
    return "This restaurant is recommended 👌👌"

def main():
    st.title('Restaurant Recommendation Web App')
    customer_id = st.text_input("Enter customer_id")
    customer_id = OrdinalEncoder().fit_transform([[customer_id]])
    gender = st.selectbox('Select Gender:', ['Male','Female','Other'])
    if gender == 'Male':
        gender = 0
    elif gender == 'Female':
        gender = 1
    else:
        gender = 2
    location_number = st.slider('What is the location number', 0,11,1)
    location_type = st.selectbox('Select location type:', ['Home','Work','Other'])
    if location_type == 'Home':
        location_type = 0
    elif location_type == 'Work':
        location_type = 1
    else:
        location_type = 2
    latitude_x = st.slider('What is the latitude_x', -91.0, 89.0, 0.1)
    longitude_x = st.slider('What is the longitude_x', -181.0, 179.0, 0.1)
    id = st.number_input("Enter vendor id")
    latitude_y = st.slider('What is the latitude_y', -91.0, 89.0, 0.1)
    longitude_y = st.slider('What is the longitude_y', -181.0, 179.0, 0.1)
    vendor_category_en = st.selectbox('Select Vendor Category:', ['Restaurants','Sweets & Bakes'])
    if vendor_category_en == 'Male':
        vendor_category_en = 0
    else:
        vendor_category_en = 1
    delivery_charge = st.selectbox('Select Delivery Charge:', [0.7,0.0])
    serving_distance = st.slider('What is the serving distance', 0.0, 50.0, step=0.5)
    bins = [0,6,10,12,50]
    labels=[1,2,3,4]
    serving_distance  = pd.cut([serving_distance], bins=bins, labels=labels, include_lowest=True)[0]
    is_open = st.selectbox('Select Open Status:', [1,0])
    prepration_time = st.slider('What is the Preparation Time', 0, 60, step=1)
    bins = [0,10,15,20,60]
    labels=[1,2,3,4]
    preparation_time  = pd.cut([prepration_time], bins=bins, labels=labels, include_lowest=True)[0]
    discount_percentage = st.slider('What is the discount_percentage', 0, 15, step=1)
    status_y = st.selectbox('Select Status:', [1,0])
    verified_y = st.selectbox('Select verified Status:', [1,0])
    rank= st.selectbox('Select rank:', [1,0])
    vendor_rating = st.slider('What is the vendor rating', 0.0, 5.0, step=0.5)
    bins = [0,3.5,4,4.5,5]
    labels=[1,2,3,4]  
    vendor_rating  = pd.cut([vendor_rating], bins=bins, labels=labels, include_lowest=True)[0]
    device_type= 0
    distance= math.sqrt( (longitude_y - longitude_x)**2 + (latitude_x-latitude_y)**2 )
        
   
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = predict_result([customer_id, gender, location_number, location_type,
       latitude_x, longitude_x, id, latitude_y, longitude_y,
       vendor_category_en, delivery_charge, serving_distance, is_open,
       prepration_time, discount_percentage, status_y, verified_y,
       rank, vendor_rating, device_type, distance, preparation_time])
        
        
    st.success(diagnosis)
if __name__ == '__main__':
    main()
