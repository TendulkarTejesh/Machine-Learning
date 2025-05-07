# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:17:48 2025

@author: HP
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import joblib

model = tf.keras.models.load_model('C:\\Tendulkar Docs\\Machine Learning\\Deployment\\Construction_Cost_Estimation_Trained_Model\\Construction_Cost_Estimation_Trained.keras')
bundle = joblib.load('C:\\Tendulkar Docs\\Machine Learning\\Deployment\\bundle_pack.joblib')
scaler = bundle['preprocessor']
column_names = bundle['feature_columns']

def Construction_Cost_Estimation(input_data):
    
    ip_df = pd.DataFrame([input_data], columns = column_names)
    ip_df = ip_df.astype({'Material_Cost' : 'int', 
                          'Labor_Cost' : 'int', 
                          'Profit_Rate' : 'int', 
                          'Discount_or_Markup' : 'int'})
    scaled_ip = scaler.transform(ip_df)
    predictions = model.predict(scaled_ip).ravel()
    return f"The Estimated Cost of your construction is {round(float(predictions[0]), 2)}"

def main():
    
    st.title('Construction Cost Estimation Web App')
    
    Material_Cost = st.text_input('Enter the Cost of the Raw Materials')
    Labor_Cost = st.text_input('Enter the Labour Cost')
    Profit_Rate = st.text_input('Enter the Profit Percentage')
    Discount_or_Markup = st.text_input('Enter the Total Discount Applied')
    
    
    Estimated_Cost = ''
    
    if st.button('Construction Cost Estimator'):
        try:
            input_features = [float(Material_Cost),
                              float(Labor_Cost),
                              float(Profit_Rate),
                              float(Discount_or_Markup)]
            Estimated_Cost = Construction_Cost_Estimation(input_features)
        
            st.success(Estimated_Cost)
        except ValueError:
            st.error('Please enter valid numerical values for all fields')

if __name__ == '__main__':
    main()
        