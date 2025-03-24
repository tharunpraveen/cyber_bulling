import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load CNN models without classification layers
@st.cache_resource
def load_feature_extractor(model_name):
    if model_name == "VGG16":
        return VGG16(weights="imagenet", include_top=False, pooling="avg")
    else:
        return ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load ML classifiers
@st.cache_resource
def load_ml_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Streamlit UI
st.title("Feature Extraction & Classification App")
st.write("Upload an image, extract features using CNN, and classify using ML models.")

# Select feature extractor and ML model
cnn_model_choice = st.selectbox("Choose a CNN feature extractor", ["VGG16", "ResNet50"])
ml_model_choice = st.selectbox("Choose an ML classifier", ["Logistic Regression", "Random Forest", "SGD Classifier"])

# Load selected models
cnn_model = load_feature_extractor(cnn_model_choice)

ml_models = {
    "Logistic Regression": load_ml_model("logistic_regression.pkl"),
    "Random Forest": load_ml_model("random_forest.pkl"),
    "SGD Classifier": load_ml_model("sgd_classifier.pkl")
}

ml_model = ml_models[ml_model_choice]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Preprocess image
def preprocess_image(img, model_choice):
    img = img.resize((224, 224))  # Resize for CNN models
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return vgg_preprocess(img_array) if model_choice == "VGG16" else resnet_preprocess(img_array)

# Predict button
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract features
    img_array = preprocess_image(img, cnn_model_choice)
    features = cnn_model.predict(img_array).flatten()  # Extract features

    # Classify using ML model
    prediction = ml_model.predict([features])[0]

    st.write(f"### Prediction: {prediction}")
