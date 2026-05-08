import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🫀 Heart Disease Prediction (Clinical Data)")
st.write("Enter patient details to predict heart disease risk")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return pickle.load(open("random_forest_model.pkl", "rb"))

model = load_model()

# -------------------- SHOW DATA DISTRIBUTION --------------------
st.subheader("📊 Dataset Insights")

data = {
    "Column": [
        "Sex", "Chest Pain Type", "FBS", "RestECG",
        "Exercise Angina", "ST Slope", "CA", "Thal", "Target"
    ],
    "Common Values": [
        "0,1", "0,1,2,3", "0,1", "0,1,2",
        "0,1", "0,1,2", "0,1,2,3,4", "0,1,2,3", "0,1"
    ]
}

st.table(pd.DataFrame(data))

# -------------------- INPUT FORM --------------------
st.subheader("🧾 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)

    sex = st.selectbox("Sex", {
        0: "Female",
        1: "Male"
    }.keys(), format_func=lambda x: {0: "Female", 1: "Male"}[x])

    cp = st.selectbox("Chest Pain Type", {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }.keys(), format_func=lambda x: {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }[x])

    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

    fbs = st.selectbox("Fasting Blood Sugar > 120", {
        0: "No",
        1: "Yes"
    }.keys(), format_func=lambda x: {0: "No", 1: "Yes"}[x])

with col2:
    restecg = st.selectbox("Resting ECG", {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "Left Ventricular Hypertrophy"
    }.keys(), format_func=lambda x: {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "LV Hypertrophy"
    }[x])

    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

    exang = st.selectbox("Exercise Induced Angina", {
        0: "No",
        1: "Yes"
    }.keys(), format_func=lambda x: {0: "No", 1: "Yes"}[x])

    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

    slope = st.selectbox("ST Slope", {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }.keys(), format_func=lambda x: {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }[x])

    # ✅ FIXED (added 4)
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3,4])

    thal = st.selectbox("Thalassemia", {
        0: "Unknown",
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }.keys(), format_func=lambda x: {
        0: "Unknown",
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x])

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error("🔴 High Risk of Heart Disease")
    else:
        st.success("🟢 Low Risk of Heart Disease")

    st.metric("Risk Probability", f"{probability:.2f}")

    # -------------------- INTERPRETATION --------------------
    st.subheader("🧠 Interpretation")

    if probability > 0.7:
        st.error("High probability — immediate medical attention recommended")
    elif probability > 0.4:
        st.warning("Moderate risk — further testing advised")
    else:
        st.success("Low risk — maintain healthy lifestyle")