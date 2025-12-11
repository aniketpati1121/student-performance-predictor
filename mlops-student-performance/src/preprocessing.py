import streamlit as st
import joblib
import numpy as np
import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "model", "student_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "artifacts", "scaler.pkl")

# LOAD OBJECTS 
scaler = joblib.load(SCALER_PATH)

#  UI
st.title("🎓 Student Performance Predictor")
st.write("Fill the student details to predict the score")

# User Inputs
gender = st.selectbox("Gender", ["male", "female"])
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0)
parent_education = st.selectbox("Parent Education", ["high_school", "bachelor", "master"])

# ENCODING 
gender_encoded = 1 if gender == "male" else 0

parent_education_map = {
    "high_school": 0,
    "bachelor": 1,
    "master": 2
}
parent_education_encoded = parent_education_map[parent_education]

# Create input array (must match training feature order)
input_data = np.array([[gender_encoded, study_hours, parent_education_encoded]])

# PREDICT 
if st.button("Predict Performance"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        st.success(f"Predicted Student Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
