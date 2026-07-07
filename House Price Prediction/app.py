import streamlit as st
import tensorflow as tf
import numpy as np

st.title("🏠 House Rent Predictor")

st.write("Enter the house details below to predict the monthly rent.")

# Inputs
bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)
size = st.number_input("Size (sq.ft.)", min_value=100, max_value=10000, step=10)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
floor = st.number_input("Floor Number", min_value=0, max_value=100, step=1)

# Load TensorFlow model
model = tf.keras.models.load_model("house_Price_model.h5")

if st.button("Predict Rent"):


    input_data = [[
        bhk,
        size,
        bathroom,
        floor
    ]]

    prediction = model.predict(np.array(input_data).reshape(1, -1))

    rent = prediction[0][0]

    st.success(f"Predicted House Rent: ₹{rent:,.0f} per month")