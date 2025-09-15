import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("breast_cancer_model.pkl")

st.title("🧪 Breast Cancer Prediction Test")

st.markdown("Enter the input features below to predict whether the tumor is Benign or Malignant.")

# Define feature names (30 total)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Create input fields dynamically
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0)
    input_data.append(value)

# Prediction
if st.button("Predict"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("🔴 Prediction: Malignant (Cancerous)")
    else:
        st.success("🟢 Prediction: Benign (Non-Cancerous)")
