# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:50:44 2025

@author: HP
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image

model = tf.keras.models.load_model('C:\\Tendulkar Docs\\Machine Learning\\Deployment\\Construction_Crack_Detection\\construction_crack_detection.keras')

def crack_detection(input_data):
    input_img = Image.open(input_data)
    if input_img.mode != 'RGB':
        input_img = input_img.convert('RGB')
    image_array = image.img_to_array(input_img)
    image_array = np.expand_dims(image_array, axis = 0)
    
    prediction = model.predict(image_array)
    
    output = (prediction > 0.5 ).astype('int32')
    #op_label = 'Cracked ' if output[0][0] == 1 else 'Non-Cracked'
    return "The input image has cracks in it" if output[0][0] == 1 else "The input image has no cracks in it"

def main():
    st.title('Crack Detection Application')
    st.write('Upload an image of a concrete surface to check for cracks')
    uploaded_file = st.file_uploader("Choose an image to upload", type = ["jpeg", "jpg", "png"])
    
    Feedback = ''
    
    if st.button('Crack Detection on a concrete surface'):
        try:
            if uploaded_file is not None:
                Feedback = crack_detection(uploaded_file)
            
            st.success(Feedback)
        
        except ValueError:
            
            st.error("Please upload the image in the given format")
            
            
if __name__ == '__main__':
    main()
            
    