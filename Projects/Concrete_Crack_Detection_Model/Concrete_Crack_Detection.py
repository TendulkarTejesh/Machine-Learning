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

@st.cache_resource
def loaded_model():
    return tf.keras.model.load_model('Projects/Concrete_Crack_Detection_Model/crack_detection_model.keras', compile = False)
    
model = loaded_model()

def crack_detection(input_data):
    input_img = image.load_img(input_data, target_size = (227, 227))
    if input_img.mode != 'RGB':
        input_img = input_img.convert('RGB')
    image_array = image.img_to_array(input_img) / 255.0
    image_array = np.expand_dims(image_array, axis = 0)
    
    
    output = model.predict(image_array)
    
    prediction = (output > 0.5 ).astype('int32').flatten()
    #op_label = 'Cracked ' if output[0][0] == 1 else 'Non-Cracked'
    return "The input image has cracks in it" if prediction[0] == 1 else "The input image has no cracks in it"

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
            
    
