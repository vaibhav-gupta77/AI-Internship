import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Wine Quality Checker", page_icon="🍷")

st.title("🍷 Wine Quality Checker")
st.write("Enter the wine properties to predict the quality class.")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "wine_quality_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


model = load_model()

with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed acidity", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
    volatile_acidity = st.number_input("Volatile acidity", min_value=0.0, max_value=5.0, value=0.3, step=0.1)
    citric_acidity = st.number_input("Citric acid", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.2, step=0.01)
    residual_sugar = st.number_input("Residual sugar", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide", min_value=0.0, max_value=72.0, value=15.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide", min_value=0.0, max_value=289.0, value=50.0, step=1.0)
    density = st.number_input("Density", min_value=0.0, max_value=1.5, value=0.996, step=0.001)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=0.0, max_value=15.0, value=10.0, step=0.1)

    submitted = st.form_submit_button("Predict Quality")

if submitted:
    features = np.array([
        [fixed_acidity, volatile_acidity, citric_acidity, pH, residual_sugar, chlorides,
         free_sulfur_dioxide, total_sulfur_dioxide, density, sulphates, alcohol]
    ], dtype=float)
    prediction = model.predict(features, verbose=0)
    quality = int(np.argmax(prediction) + 3)
    st.success(f"Predicted wine quality class: {quality}")