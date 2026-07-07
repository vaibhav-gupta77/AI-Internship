import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.write("Enter the house details below to estimate the price.")


@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("house_price_model.joblib")
    if not model_path.exists():
        st.error("Model file not found. Please make sure house_price_model.joblib is in the same folder.")
        return None
    return joblib.load(model_path)


model = load_model()

if model is not None:
    bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)
    size = st.number_input("Size (sq.ft.)", min_value=100, max_value=10000, step=10)
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
    floor = st.number_input("Floor Number", min_value=0, max_value=100, step=1)

    if st.button("Predict Price"):
        input_data = np.array([[bhk, size, bathroom, floor]], dtype=float)
        prediction = model.predict(input_data)[0]
        price = float(prediction)
        st.success(f"Estimated Price: ₹{price:,.0f}")