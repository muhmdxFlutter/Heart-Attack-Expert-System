import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from inference_mapper import load_schema, prepare_patient_vector

# --- 1. Page Configuration ---
st.set_page_config(page_title="Heart Attack Expert System", layout="wide")

# --- 2. Load Resources ---
@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'expert_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
    schema_path = os.path.join(BASE_DIR, 'models', 'model_schema.json')
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        schema = load_schema(schema_path)
        
        return None, model, scaler, schema
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

error_msg, expert_model, scaler, schema = load_resources()

if error_msg:
    st.error(error_msg)
    st.stop()

# --- 3. User Interface ---
st.title("❤️ Heart Attack Expert System ")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    age = st.selectbox("Age Category", ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    st.header("🏃 Lifestyle & Health")
    gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
    smoker = st.selectbox("Smoker Status", ["Never smoked", "Former smoker", "Current smoker - now smokes every day", "Current smoker - now smokes some days"])
    alcohol = st.selectbox("Alcohol Drinker?", ["Yes", "No"])
    physical_act = st.selectbox("Physical Activities", ["Yes", "No"])
    sleep_hours = st.slider("Sleep Hours", 1, 24, 7)

with col3:
    st.header("📋 Medical History")
    angina = st.selectbox("Had Angina (Chest Pain)?", ["No", "Yes"])
    stroke = st.selectbox("Had Stroke?", ["No", "Yes"])
    diabetic = st.selectbox("Diabetic Status", ["No", "Yes", "No, pre-diabetes or borderline diabetes", "Yes, but only during pregnancy (female)"])
    kidney = st.selectbox("Kidney Disease?", ["No", "Yes"])
    diff_walking = st.selectbox("Difficulty Walking?", ["No", "Yes"])

st.divider()

col_btn, col_res = st.columns([1, 2])

with col_btn:
    st.write(" ") # Spacing
    st.write(" ")
    run_analysis = st.button("🚀 Run AI Analysis", use_container_width=True)

with col_res:
    if run_analysis:
        # Calculate weight dynamically based on BMI and baseline height (1.7m) to prevent physical impossibilities
        computed_weight = bmi * (1.7 ** 2)
        
        user_inputs = {
            'AgeCategory': age,
            'Sex': sex,
            'BMI': bmi,
            'WeightInKilograms': computed_weight,
            'HeightInMeters': 1.7,
            'GeneralHealth': gen_health,
            'SmokerStatus': smoker,
            'AlcoholDrinkers': alcohol,
            'PhysicalActivities': physical_act,
            'SleepHours': float(sleep_hours),
            'HadAngina': angina,
            'HadStroke': stroke,
            'HadDiabetes': diabetic,
            'HadKidneyDisease': kidney,
            'DifficultyWalking': diff_walking
        }
        
        input_scaled, _ = prepare_patient_vector(user_inputs, schema, scaler)
        prob = expert_model.predict(input_scaled)[0][0]
        
        st.metric("Heart Attack Risk Probability", f"{prob*100:.2f}%")
        
        if prob > 0.5:
            st.error("🚨 HIGH RISK: Please consult a doctor immediately.")
        else:
            st.success("✅ LOW RISK: Keep maintaining a healthy lifestyle.")

st.divider()
st.caption("Developed by: Mohamed Mustafa AbdelAziz | O6U")