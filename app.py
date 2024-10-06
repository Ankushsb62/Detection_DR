import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
from PIL import Image
import io

# Load the trained models
@st.cache_resource
def load_models():
    binary_model = load_model('B-DR-MobileNet2V.h5')
    multi_model = load_model('M-DR-MobileNet2V (2).h5')
    return binary_model, multi_model

binary_model, multi_model = load_models()

# Define class names
binary_class_names = np.array(['DR', 'No_DR'])
multi_class_names = np.array(['Mild', 'Moderate', 'Proliferate_Dr', 'Severe'])

# Helper function to process the image
def process_image(file):
    image = Image.open(file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Diabetic Retinopathy Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)
    
    st.subheader('Binary Classification')
    binary_prediction = binary_model.predict(image)
    binary_class = binary_class_names[np.argmax(binary_prediction)]
    st.write(f'Binary Prediction: {binary_class}')
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if binary_class == 'DR':
        st.subheader('Multi-class Classification')
        multi_prediction = multi_model.predict(image)
        multi_class = multi_class_names[np.argmax(multi_prediction)]
        st.write(f'Multi-class Prediction: {multi_class}')
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
