import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model.h5")

st.title("🚀 Simple TensorFlow + Streamlit App")

st.write("This app predicts values for the equation **y = 2x + 1** (learned by the model).")

# User input
x_val = st.number_input("Enter a number", value=0.0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[x_val]]))
    st.success(f"Predicted value: {prediction[0][0]:.2f}")
