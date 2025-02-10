import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="🌍 Ecosystem Health Predictor",
    page_icon="🌿",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('gaussian_nb_model.pkl')

# Main title with emoji
st.title("🌍 Ecosystem Health Prediction System")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📊 Input Parameters")

# Input features with appropriate symbols
water_quality = st.sidebar.slider(
    "💧 Water Quality",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

air_quality = st.sidebar.slider(
    "💨 Air Quality Index",
    min_value=0.0,
    max_value=200.0,
    value=100.0
)

biodiversity = st.sidebar.slider(
    "🦋 Biodiversity Index",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

vegetation = st.sidebar.slider(
    "🌱 Vegetation Cover",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

soil_ph = st.sidebar.slider(
    "🪴 Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=7.0
)

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Current Measurements")
    metrics_df = pd.DataFrame({
        'Metric': ['Water Quality', 'Air Quality', 'Biodiversity', 'Vegetation', 'Soil pH'],
        'Value': [water_quality, air_quality, biodiversity, vegetation, soil_ph],
        'Symbol': ['💧', '💨', '🦋', '🌱', '🪴']
    })
    st.dataframe(metrics_df, hide_index=True)

# Make prediction
def predict_health():
    model = load_model()
    features = np.array([[water_quality, air_quality, biodiversity, vegetation, soil_ph]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    return prediction, probability

if st.sidebar.button("🔍 Predict Ecosystem Health"):
    prediction, probability = predict_health()
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Map numeric prediction to labels with symbols
        health_labels = {
            0: "✅ Healthy",
            1: "⚠️ At Risk",
            2: "❌ Degraded"
        }
        
        st.markdown(f"""
        ### Status: {health_labels[prediction]}
        """)
        
        # Display prediction probabilities
        st.markdown("### Confidence Levels:")
        prob_df = pd.DataFrame({
            'Status': ['Healthy', 'At Risk', 'Degraded'],
            'Probability': probability,
            'Symbol': ['✅', '⚠️', '❌']
        })
        
        for _, row in prob_df.iterrows():
            st.markdown(f"{row['Symbol']} {row['Status']}: {row['Probability']*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("### 📝 About")
st.markdown("""
This application uses a Gaussian Naive Bayes model to predict ecosystem health based on various environmental parameters.
- 💧 Water Quality: Measures the cleanliness and safety of water bodies
- 💨 Air Quality Index: Indicates the level of air pollution
- 🦋 Biodiversity Index: Represents the variety of species in the ecosystem
- 🌱 Vegetation Cover: Percentage of area covered by plants
- 🪴 Soil pH: Acidity or alkalinity of the soil
""")