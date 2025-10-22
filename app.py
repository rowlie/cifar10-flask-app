import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

st.title("CIFAR-10 Image Classification")
st.write("Upload an image and choose a model to classify it.")

classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load models and cache them to speed up reloads
@st.cache_resource
def load_models():
    baseline = load_model("best_baseline.h5")
    vgg16 = load_model("vgg16_transfer.h5")
    return {"Baseline": baseline, "VGG16_Transfer": vgg16}

models = load_models()
model_name = st.selectbox("Choose Model:", options=["Baseline", "VGG16_Transfer"])
uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    model = models.get(model_name, None)
    if model is not None:
        predictions = model.predict(img_array)
        pred_class = classnames[np.argmax(predictions)]
        st.markdown(f"**Selected Model:** {model_name}")
        st.markdown(f"**Predicted Class:** {pred_class}")
    else:
        st.error("Model not found or failed to load.")
