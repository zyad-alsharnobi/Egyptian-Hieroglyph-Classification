import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# Load models (make sure these paths are correct)
cnn_test_model = tf.keras.models.load_model("cnnbest_model.keras")
rns_test_model = tf.keras.models.load_model("rnsbest_model.keras")
dns_test_model = tf.keras.models.load_model("dnsbest_model.keras")
efn_test_model = tf.keras.models.load_model("efnbest_model.keras")

# Dictionary to map model names to model objects
models = {
    "CNN": cnn_test_model,
    "ResNet": rns_test_model,
    "DenseNet": dns_test_model,
    "EfficientNet": efn_test_model
}
import pandas as pd

# Load class names from annotations CSV
def get_class_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['class'].unique())
    class_names = {index: class_name for index, class_name in enumerate(unique_classes)}
    return class_names

csv_path = "_annotations.csv" 
class_names = get_class_names_from_csv(csv_path)




# Function to preprocess image for prediction
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def make_prediction(model, image):
    preds = model.predict(image)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds, axis=1)[0]
    class_name = class_names.get(class_idx, "Unknown")  # Get class name from mapping
    return class_name, confidence

# Streamlit app interface
st.title("Egyptian Hieroglyphs Classification")

# Image upload
uploaded_file = st.file_uploader("Upload an image of a hieroglyph", type=["jpg", "jpeg", "png"])

# Model selection
model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))

# Predict button and result display
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    if st.button("Classify"):
        st.write("Processing the image and making predictions...")

        # Preprocess and make prediction
        model = models[model_choice]
        processed_image = preprocess_image(image)
        class_name, confidence = make_prediction(model, processed_image)

        # Display the result
        st.write(f"Predicted Class: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")
