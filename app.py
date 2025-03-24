import streamlit as st
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# -----------------------
# Load ML Models from Hugging Face
# -----------------------

@st.cache_resource
def load_ml_model(model_name):
    model_path = hf_hub_download(repo_id="https://huggingface.co/Pandu1729/models", filename=model_name)
    return joblib.load(model_path)

ml_models = {
    "Logistic Regression": load_ml_model("logistic_regression.pkl"),
    "Random Forest": load_ml_model("random_forest.pkl"),
    "SGD Classifier": load_ml_model("sgd_classifier.pkl")
}

# -----------------------
# Load Pre-trained CNN Models for Feature Extraction
# -----------------------

@st.cache_resource
def load_cnn_models():
    vgg_model = VGG16(weights="imagenet", include_top=False)
    resnet_model = ResNet50(weights="imagenet", include_top=False)
    return vgg_model, resnet_model

vgg_model, resnet_model = load_cnn_models()

# -----------------------
# Feature Extraction Function
# -----------------------

def extract_features(img_path, model, preprocess_func):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)  # Apply preprocessing

    features = model.predict(img_array)
    return features.flatten()  # Convert to 1D vector

# -----------------------
# Streamlit UI
# -----------------------

st.title("Cyberbullying Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Feature Extraction Using VGG16 & ResNet50
    vgg_features = extract_features(uploaded_file, vgg_model, vgg_preprocess)
    resnet_features = extract_features(uploaded_file, resnet_model, resnet_preprocess)

    # Combine VGG16 and ResNet50 Features
    combined_features = np.concatenate((vgg_features, resnet_features))

    # -----------------------
    # Make Predictions Using All Models
    # -----------------------

    st.subheader("Model Predictions:")

    predictions = {}
    for model_name, model in ml_models.items():
        pred = model.predict([combined_features])[0]  # Get prediction
        predictions[model_name] = pred

    # Display results in a table
    st.table(predictions)
