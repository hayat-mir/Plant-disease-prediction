import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Function to download the model from Hugging Face
def download_model(url, model_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'plant_disease_prediction_model.h5')
class_indices_path = os.path.join(working_dir, 'class_indices.json')

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    hugging_face_url = "https://huggingface.co/hayat52-m/plant-disease-prediction/resolve/main/plant_disease_prediction_model.h5"
    try:
        download_model(hugging_face_url, model_path)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class indices
class_indices = json.load(open(class_indices_path))

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit app
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
