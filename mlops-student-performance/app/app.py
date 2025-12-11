import streamlit as st
import joblib
import numpy as np
import os

# Paths
MODEL_PATH = os.path.join("artifacts", "model", "student_model.pkl")
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# App title
st.title("Student Performance Predictor")
st.write("Enter student details to predict performance:")

# User inputs
gender = st.selectbox("Gender", ["male", "female"])
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
parent_education = st.selectbox("Parent Education", ["high_school", "bachelor", "master"])
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)

# Encode categorical inputs
gender_encoded = 1 if gender == "male" else 0
parent_education_map = {"high_school": 0, "bachelor": 1, "master": 2}
parent_education_encoded = parent_education_map[parent_education]

# Create input array (matches trained model features)
input_data = np.array([[gender_encoded, study_hours, parent_education_encoded, previous_score]])

# Predict button
if st.button("Predict Performance"):
    # Scale input
    scaled_input = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(scaled_input)
    # Display result
    st.success(f"Predicted Student Score: {prediction[0]:.2f}")
