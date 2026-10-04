import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/crop_recommendation_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 Crop Recommendation System")

st.write(
    "This Machine Learning system recommends a suitable crop "
    "based on soil and environmental conditions."
)

st.divider()

# Input section
st.subheader("Enter Crop Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=150.0,
        value=50.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=150.0,
        value=50.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=210.0,
        value=50.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=300.0,
        value=100.0
    )

st.divider()

# Recommendation button
if st.button("🌾 Recommend Crop", use_container_width=True):

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    prediction = model.predict(input_data)

    crop = prediction[0].capitalize()

    st.success(f"🌱 Recommended Crop: **{crop}**")

crop_info = {
    "Rice": "Rice generally requires warm temperatures, high humidity and sufficient rainfall.",
    "Maize": "Maize grows well in warm conditions with suitable soil nutrients and moderate rainfall.",
    "Wheat": "Wheat generally prefers cool to moderate temperatures and well-drained soil.",
    "Cotton": "Cotton grows well in warm temperatures with suitable rainfall and soil conditions.",
    "Mango": "Mango is a tropical fruit crop that generally prefers warm temperatures and suitable rainfall.",
    "Banana": "Banana grows well in warm, humid conditions with adequate water availability.",
    "Apple": "Apple generally prefers cooler temperatures and suitable soil conditions.",
    "Chickpea": "Chickpea generally grows well in relatively dry conditions and suitable soil nutrients.",
    "Kidneybeans": "Kidney beans require suitable moisture, temperature and soil nutrients.",
    "Grapes": "Grapes generally grow well in warm conditions with suitable soil and moderate rainfall."
}

if crop in crop_info:
    st.info(f"ℹ️ **About {crop}:** {crop_info[crop]}")
else:
    st.info(
        "ℹ️ The recommendation is generated using the trained "
        "Random Forest Machine Learning model."
    )

# Project information
st.divider()

st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Test Accuracy", "99.54%")

with col3:
    st.metric("Input Features", "7")