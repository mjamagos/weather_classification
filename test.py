import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Weather AI", layout="centered")
st.title("Weather Classification App")
st.write("Upload a photo and the AI will tell you if it's Cloudy, Rainy, or a Thunderstorm.")

class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)  # ignore legacy arg
        return super().from_config(config)
    
# --- LOAD YOUR MODEL ---
@st.cache_resource  # This stops the app from reloading the heavy model every time you click
def load_my_ai():
    model = tf.keras.models.load_model("keras_Model.h5", compile=False)
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    return model, class_names

model, class_names = load_my_ai()

# --- THE UPLOADER (No more hardcoding!) ---
uploaded_file = st.file_uploader("Choose a weather image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Open and show the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", use_container_width=True)
    
    # 2. Process the image (The Teachable Machine Logic)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 3. Predict
    with st.spinner('AI is thinking...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # 4. Display Results
    st.success(f"Prediction: **{class_name[2:]}**")
    st.metric(label="Confidence Score", value=f"{round(confidence_score * 100, 2)}%")