import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Model (update path if needed)
# -----------------------------
model = joblib.load("model.pkl")   # <-- make sure your model is saved as model.pkl

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")

# -----------------------------
# Input Fields
# -----------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1, max_value=120)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Built with ❤️ using Streamlit")