import streamlit as st
import joblib
import numpy as np
from PIL import Image

knn = joblib.load("knn_model.pkl")
pca = joblib.load("pca_model.pkl")

st.set_page_config(page_title="Handwritten Digit Recognition")

st.title("Handwritten Digit Recognition (MNIST)")
st.write("Upload a handwritten digit image")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    
    st.image(image, caption="Uploaded Image", width=150)

    img = np.array(image).reshape(1, -1) / 255.0
    img_pca = pca.transform(img)

    prediction = knn.predict(img_pca)

    st.success(f"Predicted Digit: {prediction[0]}")
