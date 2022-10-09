

import xgboost 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.preprocessing import OrdinalEncoder
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
loaded_model = pickle.load(open('model.pkl', 'rb'))

@st.cache
def predict_result(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction == 0 :
        return "I Would not recommended this restaurant 👎🙅‍♂️"
    return "I Would recommended restaurant 👌👌"

def main():
    st.title('Restaurant Recommendation Web App')
    customer_id = 0
    # customer_id = st.text_input("Enter customer_id")
    # customer_id = OrdinalEncoder().fit_transform([[customer_id]])
    gender = st.radio('Select Your Gender:', ['Male','Female','Other'],horizontal=True)
    if gender == 'Male':
        gender = 0
    elif gender == 'Female':
        gender = 1
    else:
        gender = 2
    location_number = st.slider('What is the location number?', 0,11,1)
    location_type = st.radio('Select your location type:', ['Home','Work','Other'],horizontal=True)
    if location_type == 'Home':
        location_type = 0
    elif location_type == 'Work':
        location_type = 1
    else:
        location_type = 2
    latitude_x = st.slider('What is your latitude co-ordinate?', -91.0, 89.0, 0.1)
    longitude_x = st.slider('What is your longitude co-ordinate?', -181.0, 179.0, 0.1)
    id = st.number_input("Enter vendor id",min_value=0)
    latitude_y = st.slider("What is the vendor's latitude co-ordinate?", -91.0, 89.0, 0.1)
    longitude_y = st.slider("What is the vendor's longitude co-ordinate?", -181.0, 179.0, 0.1)
    vendor_category_en = st.radio('Select Vendor Category:', ['Restaurants','Sweets & Bakes'],horizontal=True)
    vendor_category_en = 1 if vendor_category_en == 'Restaurants' else 0
    delivery_charge = st.radio('Delivery Charge Applicable?:', ["Yes","No"],horizontal=True)
    delivery_charge = 1 if delivery_charge =="Yes" else 0
    serving_distance = st.slider('What is the serving distance?', 0.0, 50.0, step=0.25)
    bins = [0,6,10,12,50]
    labels=[1,2,3,4]
    serving_distance  = pd.cut([serving_distance], bins=bins, labels=labels, include_lowest=True)[0]
    is_open = st.radio('Is vendor Open? :',  ["Yes","No"],horizontal=True)
    is_open = 1 if is_open =="Yes" else 0
    prepration_time = st.slider('What is the Preparation Time for the order?', 0, 60, step=1)
    bins = [0,10,15,20,60]
    labels=[1,2,3,4]
    preparation_time  = pd.cut([prepration_time], bins=bins, labels=labels, include_lowest=True)[0]
    discount_percentage = st.slider('What is the discount_percentage', 0, 15, step=1)
    status_y = st.radio('Customer status:', ["Active","Inactive"], horizontal=True)
    status_y = 1 if status_y =="Active" else 0
    verified_y = st.radio('Are you Verified?:', ["Yes","No"],horizontal=True)
    verified_y = 1 if verified_y =="Yes" else 0
    rank= st.radio('Is vendor ranked:',  ["Yes","No"],horizontal=True)
    rank = 1 if rank =="Yes" else 0
    vendor_rating = st.slider('What is the vendor rating', 0.0, 5.0, step=0.5)
    bins = [0,3.5,4,4.5,5]
    labels=[1,2,3,4]  
    vendor_rating  = pd.cut([vendor_rating], bins=bins, labels=labels, include_lowest=True)[0]
    device_type= 0
    distance= math.sqrt( (longitude_y - longitude_x)**2 + (latitude_x-latitude_y)**2 )
        
   
    Result = ''
    
    # creating a button for Prediction
    if st.button('Would you recommend? 🤔'):
        st.balloons()
        Result = predict_result([customer_id, gender, location_number, location_type,
       latitude_x, longitude_x, id, latitude_y, longitude_y,
       vendor_category_en, delivery_charge, serving_distance, is_open,
       prepration_time, discount_percentage, status_y, verified_y,
       rank, vendor_rating, device_type, distance, preparation_time])
        if "not" in Result:
            st.error(Result)        
        else:
            st.success(Result)
        

       
        
        
    
if __name__ == '__main__':
    main()
