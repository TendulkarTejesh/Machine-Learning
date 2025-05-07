# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:04:53 2025

@author: HP
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import streamlit as st

model = tf.keras.models.load_model('Projects/Airline_Customer_Satisfaction_Prediction_Model/Airline_Satisfaction.keras')
bundle = joblib.load('Projects/Airline_Customer_Satisfaction_Prediction_Model/encoder_bundle.joblib')
#column_names = joblib.load('Projects/Airline_Customer_Satisfaction_Prediction_Model/columns.joblib')
preprocessed = bundle['preprocessor']
label_encoder = bundle['labelencoder']
column_names = bundle['columns']

def Airline_Satisfaction(input_data):
    
    ip_df = pd.DataFrame([input_data], columns = column_names)
    ip_df = ip_df.astype({
    'Gender': 'object',
    'Customer Type': 'object',
    'Age': 'int',
    'Type of Travel': 'object',
    'Class': 'object',
    'Flight Distance': 'int',
    'Seat comfort': 'int',
    'Departure/Arrival time convenient': 'int',
    'Food and drink': 'int',
    'Gate location': 'int',
    'Inflight wifi service': 'int',
    'Inflight entertainment': 'int',
    'Online support': 'int',
    'Ease of Online booking': 'int',
    'On-board service': 'int',
    'Leg room service': 'int',
    'Baggage handling': 'int',
    'Checkin service': 'int',
    'Cleanliness': 'int',
    'Online boarding': 'int', 
    'Departure Delay in Minutes': 'int',
    'Arrival Delay in Minutes': 'float'
})
    encoded_ip = preprocessed.transform(ip_df)
    output = model.predict(encoded_ip).ravel()
    predictions = (output > 0.5).astype(int)
    Customer_Feedback = label_encoder.inverse_transform(predictions)
    return f"Our customer feels that they are {Customer_Feedback[0]} with our airline journey"
def main():
    st.title('Airline Customer Satisfaction Application')
    Gender = st.text_input('Gender of the Passenger')
    Customer_Type = st.text_input('Enter the type of the Customer: (disloyal Cusotmer/Loyal Customer)')
    Age = st.number_input('Enter the age of the Customer')
    Type_of_Travel = st.text_input('Enter the type of travel of the Customer: (Personal Travel / Business Travel)')
    Class_Travelled = st.text_input('Enter the Class in which passenger travelled: (Eco/ Eco Plus/ Business)')
    Flight_Distance = st.number_input('Enter the Distance travelled by the Customer')
    Seat_Comfort = st.number_input('Enter the rating for Seat Comfort in the Flight : (O to 5)')
    Departure_OR_Arrival_Time_Convenient = st.number_input('Enter the rating for Departure/Arrival time convenient: (O to 5)')
    Food_and_Drink = st.number_input('Enter the rating for Food and Beverages: (O to 5)')
    Gate_Location = st.number_input('Enter the rating for Gate Location: (O to 5)')
    Inflight_Wifi_Service = st.number_input('Enter the rating for Inflight wifi service: (O to 5)')
    Inflight_Entertainment = st.number_input('Enter the rating for Inflight entertainment: (O to 5)')
    Online_Support = st.number_input('Enter the rating for Online support: (O to 5)')
    Ease_of_Online_Booking = st.number_input('Enter the rating for Ease of Online Booking: (O to 5)')
    Onboard_Service = st.number_input('Enter the rating for Onboard Services: (O to 5)')
    Legroom_Service = st.number_input('Enter the rating for Leg room service: (O to 5)')
    Baggage_Handling = st.number_input('Enter the rating for Baggage Handling: (O to 5)')
    Checkin_Service = st.number_input('Enter the rating for Checkin Service: (O to 5)')
    Cleanliness = st.number_input('Enter the rating for Cleanliness in the Flight: (O to 5)')
    Online_Boarding = st.number_input('Enter the rating for Online boarding: (O to 5)')
    Departure_Delay_in_Minutes = st.number_input('Enter the Departure Delay in minutes')
    Arrival_Delay_in_Minutes = st.number_input('Enter the Arrival Delay in minutes')
    
    
    Feedback = ''
    
    if st.button('Airline Customer Satisfaction'):
        try:
            input_features = [str(Gender), str(Customer_Type), int(Age), str(Type_of_Travel), 
                              str(Class_Travelled), float(Flight_Distance), int(Seat_Comfort),
                              int(Departure_OR_Arrival_Time_Convenient), int(Food_and_Drink), 
                              int(Gate_Location), int(Inflight_Wifi_Service), int(Inflight_Entertainment), 
                              int(Online_Support), int(Ease_of_Online_Booking), int(Onboard_Service), 
                              int(Legroom_Service), int(Baggage_Handling), int(Checkin_Service), 
                              int(Cleanliness), int(Online_Boarding), float(Departure_Delay_in_Minutes), 
                              float(Arrival_Delay_in_Minutes)]
            Feedback = Airline_Satisfaction(input_features)
                                             
            
            st.success(Feedback)
        
        except ValueError:
            st.error("Please enter valid values for all fields")
    
if __name__ == '__main__':
    main()
 
