import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below")

preg = st.number_input("Pregnancies", min_value=0, value=1)
glu = st.number_input("Glucose", min_value=0, value=120)
bp = st.number_input("Blood Pressure", min_value=0, value=70)
skin = st.number_input("Skin Thickness", min_value=0, value=20)
ins = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    sample = np.array([
        [preg, glu, bp, skin, ins, bmi, dpf, age]
    ])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely Diabetic")
    else:
        st.success("✅ Patient is likely Non-Diabetic")

    st.subheader("Prediction Probability")
    st.write(f"Diabetes Risk: {probability[0][1] * 100:.2f}%")