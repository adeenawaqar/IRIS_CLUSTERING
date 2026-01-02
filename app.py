import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load the Model and Scaler ---
@st.cache_resource
def load_files():
    # Make sure scaler.pkl and kmeans_model.pkl are in your GitHub folder
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
    return scaler, kmeans

try:
    scaler, kmeans = load_files()
except Exception as e:
    st.error("Error: Model files (.pkl) not found on GitHub!")

# --- UI Setup ---
st.set_page_config(page_title="Iris Cluster Predictor")
st.title("🌸 Iris Flower Cluster Predictor")
st.write("Enter flower measurements to find its cluster.")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length", value=5.1)
    sw = st.number_input("Sepal Width", value=3.5)

with col2:
    pl = st.number_input("Petal Length", value=1.4)
    pw = st.number_input("Petal Width", value=0.2)

# --- Prediction Logic ---
if st.button("Identify Cluster"):
    if scaler and kmeans:
        # 1. Create input array
        data = np.array([[sl, sw, pl, pw]])
        
        # 2. Scale the input
        scaled_data = scaler.transform(data)
        
        # 3. Predict Cluster
        prediction = kmeans.predict(scaled_data)[0]
        
        # 4. Display Result
        st.success(f"This flower belongs to **Cluster {prediction}**")
        
        # Display Info
        species_info = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        st.info(f"Note: Cluster {prediction} usually represents Iris-{species_info.get(prediction, 'Unknown')}")
    else:
        st.error("Model not loaded correctly.")
