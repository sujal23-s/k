import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Patient Health Status Classifier")
st.write("Enter the patient's metabolic metrics below to predict status (1: Healthy Controls, 2: Patients).")

# Create input form
age = st.number_input("Age", min_value=0, max_value=120, value=50)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose", min_value=30, max_value=300, value=90)
insulin = st.number_input("Insulin", min_value=0.0, max_value=100.0, value=5.0)
homa = st.number_input("HOMA", min_value=0.0, max_value=30.0, value=1.0)
leptin = st.number_input("Leptin", min_value=0.0, max_value=100.0, value=10.0)
adiponectin = st.number_input("Adiponectin", min_value=0.0, max_value=50.0, value=10.0)
resistin = st.number_input("Resistin", min_value=0.0, max_value=100.0, value=10.0)
mcp_1 = st.number_input("MCP-1", min_value=0.0, max_value=2000.0, value=300.0)

if st.button("Predict"):
    # Prepare inputs
    input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]])
    scaled_data = scaler.transform(input_data)
    
    # Run prediction
    prediction = model.predict(scaled_data)[0]
    status = "Healthy Control (Class 1)" if prediction == 1 else "Patient (Class 2)"
    
    st.success(f"Prediction: {status}")
