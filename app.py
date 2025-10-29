import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Model
# -------------------------------
st.title(" Skin Cancer Prediction (Metadata-Based)")
st.write("Enter patient details to predict if the case is **Benign** or **Malignant**")

# Load trained model (.h5 file)
model = load_model("skin_cancer_metadata_final.h5")

# -------------------------------
# Input fields
# -------------------------------
st.header("🔹 Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=35)
smoke = st.selectbox("Smoke", [0, 1])
drink = st.selectbox("Drink", [0, 1])
background_father = st.selectbox("Background Father", [0, 1])
background_mother = st.selectbox("Background Mother", [0, 1])
pesticide = st.selectbox("Pesticide Exposure", [0, 1])
gender = st.selectbox("Gender (0=Male, 1=Female)", [0, 1])
skin_cancer_history = st.selectbox("Skin Cancer History", [0, 1])
cancer_history = st.selectbox("Cancer History", [0, 1])
has_piped_water = st.selectbox("Has Piped Water", [0, 1])
has_sewage_system = st.selectbox("Has Sewage System", [0, 1])
fitspatrick = st.number_input("Fitspatrick Type (1-6)", min_value=1, max_value=6, value=3)
region = st.number_input("Region (encoded number)", min_value=0, value=1)
itch = st.selectbox("Itch", [0, 1])
grew = st.selectbox("Grew", [0, 1])
hurt = st.selectbox("Hurt", [0, 1])
changed = st.selectbox("Changed", [0, 1])
bleed = st.selectbox("Bleed", [0, 1])
elevation = st.selectbox("Elevation", [0, 1])
biopsed = st.selectbox("Biopsed", [0, 1])
diameter_1 = st.number_input("Diameter 1 (mm)", min_value=0.0, value=5.0)
diameter_2 = st.number_input("Diameter 2 (mm)", min_value=0.0, value=6.0)

# Collect inputs
input_data = np.array([[
    age, smoke, drink, background_father, background_mother,
    pesticide, gender, skin_cancer_history, cancer_history,
    has_piped_water, has_sewage_system, fitspatrick, region,
    itch, grew, hurt, changed, bleed, elevation, biopsed,
    diameter_1, diameter_2
]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        # Standardize input
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_data)

        # Predict
        prediction = model.predict(X_scaled)
        result = " Benign (Non-Cancerous)" if prediction < 0.5 else " Malignant (Cancerous)"

        st.subheader("Prediction Result:")
        st.success(result)
        st.write(f"**Prediction Score:** {float(prediction):.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
