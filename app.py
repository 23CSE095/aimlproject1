import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# 1️⃣ Load the trained model safely
# ----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("skin_cancer_metadata_final.h5")
    return model

model = load_model()

# ----------------------------------
# 2️⃣ Page setup
# ----------------------------------
st.set_page_config(page_title="Skin Cancer Predictor", layout="centered")
st.title("🧬 Skin Cancer Prediction from Patient Metadata")

st.write("Enter patient details below to predict if the lesion is **Benign (0)** or **Malignant (1)**.")

# ----------------------------------
# 3️⃣ Input fields
# ----------------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=40)
smoke = st.selectbox("Do you smoke?", [0, 1])
drink = st.selectbox("Do you drink?", [0, 1])
background_father = st.selectbox("Father's background (0/1)", [0, 1])
background_mother = st.selectbox("Mother's background (0/1)", [0, 1])
pesticide = st.selectbox("Exposure to pesticide?", [0, 1])
gender = st.selectbox("Gender", [0, 1])  # 0=Female, 1=Male
skin_cancer_history = st.selectbox("Previous Skin Cancer History?", [0, 1])
cancer_history = st.selectbox("Cancer History?", [0, 1])
has_piped_water = st.selectbox("Has Piped Water?", [0, 1])
has_sewage_system = st.selectbox("Has Sewage System?", [0, 1])
fitspatrick = st.slider("Fitspatrick Skin Type", 1, 6, 3)
region = st.number_input("Region code (0-10)", min_value=0, max_value=10, value=1)
itch = st.selectbox("Itching?", [0, 1])
grew = st.selectbox("Growth observed?", [0, 1])
hurt = st.selectbox("Hurts?", [0, 1])
changed = st.selectbox("Changed recently?", [0, 1])
bleed = st.selectbox("Bleeding?", [0, 1])
elevation = st.selectbox("Elevation?", [0, 1])
biopsed = st.selectbox("Biopsed?", [0, 1])
diameter_1 = st.number_input("Diameter 1 (mm)", min_value=0.0, value=5.0)
diameter_2 = st.number_input("Diameter 2 (mm)", min_value=0.0, value=5.0)

# ----------------------------------
# 4️⃣ Prepare input data
# ----------------------------------
input_data = np.array([[
    age, smoke, drink, background_father, background_mother,
    pesticide, gender, skin_cancer_history, cancer_history,
    has_piped_water, has_sewage_system, fitspatrick, region,
    itch, grew, hurt, changed, bleed, elevation, biopsed,
    diameter_1, diameter_2
]])

# StandardScaler ensures normalization for numeric input
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# ----------------------------------
# 5️⃣ Prediction
# ----------------------------------
if st.button(" Predict"):
    prediction = model.predict(input_scaled)
    result = (prediction > 0.5).astype(int)[0][0]

    if result == 1:
        st.error(" The model predicts **Malignant (Cancer Detected)**")
    else:
        st.success(" The model predicts **Benign (No Cancer Detected)**")

# ----------------------------------
# 6️⃣ Footer
# ----------------------------------
st.markdown("---")
st.caption("Developed with  using TensorFlow and Streamlit")
