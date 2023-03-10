# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:53:43 2023

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the InceptionV3 model
model = load_model("model_inception.h5")

# Define the preprocess function
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = (img - 0.5) * 2.0
    img = np.expand_dims(img, axis=0)
    return img

# Define the predict function
def predict(img):
    img = preprocess(img)
    preds = model.predict(img)
    return preds


# Create the Streamlit app
st.title("Cotton Leaf Disease Prediction")

#save Results
Results=['This plant may have Bacterial Blight disease','This plant may have Curl Virus',
         'This plant can be said to be Fresh cotton leaf','This plant may have Sucking and Chewing Pest'
         ]

# Add a file upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make a prediction on the uploaded file
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    preds = predict(image)
    label_idx = np.argmax(preds)
    label =Results[label_idx]
    st.write(f"Prediction: {label}")
    
    

    
