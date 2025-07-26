import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Pass Prediction App")

# Input fields
study_time = st.slider("Study Time (hrs/day)", 1, 10)
parent_education = st.selectbox("Parent Education Level", [1, 2, 3])  # 1: High school, 2: Bachelor's, 3: Master's
gender = st.radio("Gender", ["Male", "Female"])

# Convert gender to binary
gender_val = 1 if gender == "Male" else 0

if st.button("Predict"):
    features = np.array([[study_time, parent_education, gender_val]])
    result = model.predict(features)[0]
    if result == 1:
        st.success("✅ The student is likely to Pass.")
    else:
        st.error("❌ The student is likely to Fail.")
