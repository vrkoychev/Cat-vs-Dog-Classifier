import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('Model/cat_vs_dog_model.h5')

# Streamlit page configuration
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Display the logo at the top
logo_path = "cat_logo.png"

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns
with col2:  # Place the logo in the center column
    st.image(logo_path, width=300)

# Centered Title
st.markdown("<h1 style='text-align: center;'>Is it a cat or a dog?</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Upload an image of a cat or dog, and the model will predict it!</h3>", unsafe_allow_html=True)

# Upload image section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=(128, 128))  # Resize the image to 64x64
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array

# Prediction section
if uploaded_image is not None:
    # Preprocess the uploaded image
    img = preprocess_image(uploaded_image)
    
    # Display the image
    st.image(uploaded_image, caption="Uploaded Image", width=300)
    
    # Predict the class
    prediction = model.predict(img)
    
    # Output prediction result
    if prediction[0][0] > 0.5:
        st.write("Prediction: Dog 🐶")
    else:
        st.write("Prediction: Cat 🐱")

# Add some info about the app
st.markdown("<h3 style='text-align: center;'>About this app</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app uses a Convolutional Neural Network (CNN) model to classify images as either a cat or a dog.</p>", unsafe_allow_html=True)