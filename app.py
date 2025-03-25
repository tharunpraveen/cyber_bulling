import streamlit as st
import cv2
import numpy as np
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess
import pickle

# Cache resource-intensive components
@st.cache_resource
def load_models():
    # Create feature extractors
    def create_feature_extractor(model_class, preprocess_fn):
        base_model = model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        extractor = tf.keras.Model(inputs=base_model.input, outputs=x)
        return extractor, preprocess_fn

    return {
        'vgg16': create_feature_extractor(VGG16, vgg16_preprocess),
        'vgg19': create_feature_extractor(VGG19, vgg19_preprocess),
        'resnet50': create_feature_extractor(ResNet50, resnet50_preprocess),
        'inceptionv3': create_feature_extractor(InceptionV3, inceptionv3_preprocess),
        'inceptionresnetv2': create_feature_extractor(InceptionResNetV2, inceptionresnetv2_preprocess),
        'scaler': pickle.load(open("https://huggingface.co/Pandu1729/scaler/resolve/main/scaler.pkl", "rb")),
        'logreg': pickle.load(open("https://huggingface.co/Pandu1729/logistic_regression/resolve/main/logistic_regression.pkl", "rb")),
        'rf': pickle.load(open("https://huggingface.co/Pandu1729/random_forest/resolve/main/random_forest.pkl", "rb")),
        'sgd': pickle.load(open("https://huggingface.co/Pandu1729/sgd_classifier/resolve/main/sgd_classifier.pkl", "rb"))
    }

def load_image_from_upload(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def load_image_from_url(url):
    response = requests.get(url)
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return img

def extract_features_all_models(img, models):
    img = cv2.resize(img, (224, 224))
    features_list = []
    
    # VGG16
    img_vgg16 = np.expand_dims(img.copy(), axis=0)
    img_vgg16 = models['vgg16'][1](img_vgg16)
    features_list.append(models['vgg16'][0].predict(img_vgg16, verbose=0))
    
    # VGG19
    img_vgg19 = np.expand_dims(img.copy(), axis=0)
    img_vgg19 = models['vgg19'][1](img_vgg19)
    features_list.append(models['vgg19'][0].predict(img_vgg19, verbose=0))
    
    # ResNet50
    img_resnet50 = np.expand_dims(img.copy(), axis=0)
    img_resnet50 = models['resnet50'][1](img_resnet50)
    features_list.append(models['resnet50'][0].predict(img_resnet50, verbose=0))
    
    # InceptionV3
    img_inceptionv3 = np.expand_dims(img.copy(), axis=0)
    img_inceptionv3 = models['inceptionv3'][1](img_inceptionv3)
    features_list.append(models['inceptionv3'][0].predict(img_inceptionv3, verbose=0))
    
    # InceptionResNetV2
    img_inceptionresnetv2 = np.expand_dims(img.copy(), axis=0)
    img_inceptionresnetv2 = models['inceptionresnetv2'][1](img_inceptionresnetv2)
    features_list.append(models['inceptionresnetv2'][0].predict(img_inceptionresnetv2, verbose=0))
    
    return np.concatenate(features_list, axis=1)

def main():
    st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
    st.title("Cyberbullying Image Detection")
    st.write("Upload an image or provide a URL to check for potential cyberbullying content")

    # Load models once
    models = load_models()

    # Input selection
    input_method = st.radio("Select input method:", ("Upload Image", "Image URL"))

    img = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = load_image_from_upload(uploaded_file)
    else:
        url = st.text_input("Enter Image URL:")
        if url:
            try:
                img = load_image_from_url(url)
            except:
                st.error("Error loading image from URL")

    if img is not None:
        st.image(img, channels="BGR", width=300)
        
        with st.spinner("Analyzing image..."):
            try:
                features = extract_features_all_models(img, models)
                features_scaled = models['scaler'].transform(features)
                
                # Get predictions
                pred_lr = models['logreg'].predict(features_scaled)[0]
                pred_rf = models['rf'].predict(features_scaled)[0]
                pred_sgd = models['sgd'].predict(features_scaled)[0]

                # Display results
                st.subheader("Detection Results:")
                cols = st.columns(3)
                with cols[0]:
                    st.write("**Logistic Regression:**")
                    st.success("Non-Bullying") if pred_lr == 0 else st.error("Bullying")
                with cols[1]:
                    st.write("**Random Forest:**")
                    st.success("Non-Bullying") if pred_rf == 0 else st.error("Bullying")
                with cols[2]:
                    st.write("**SGD Classifier:**")
                    st.success("Non-Bullying") if pred_sgd == 0 else st.error("Bullying")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
