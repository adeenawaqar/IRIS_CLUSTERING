import streamlit as st
import pickle
import numpy as np

# --- 1. Load the Model and Scaler ---
@st.cache_resource
def load_files():
    try:
        # Load the files you saved from Kaggle
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
        dbscan = pickle.load(open('dbscan_model.pkl', 'rb'))
        return scaler, kmeans, dbscan
    except Exception as e:
        return None, None, None

scaler, kmeans, dbscan = load_files()

# --- 2. UI Setup ---
st.set_page_config(page_title="Iris Clustering App", layout="centered")
st.title("🌸 Iris Flower Cluster Predictor")
st.write("Enter flower measurements to identify its cluster and species.")

# --- 3. Sidebar: DBSCAN Analysis ---
st.sidebar.header("Model Comparison")
st.sidebar.write("Performance from Training Phase:")

# Silhouette scores based on your training
st.sidebar.info(f"""
- **K-Means Score:** 0.4590
- **DBSCAN Score:** 0.3492
""")

st.sidebar.warning("Note: K-Means is used for prediction. DBSCAN is shown for comparison only.")

# --- 4. Main Input Fields ---
st.subheader("Input Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1)
    sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1)

with col2:
    pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1)
    pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)

# --- 5. Prediction Logic ---
if st.button("Identify Cluster"):
    if scaler is not None and kmeans is not None:
        # Create input array
        input_data = np.array([[sl, sw, pl, pw]])
        
        # Scale the data using saved scaler
        scaled_input = scaler.transform(input_data)
        
        # Predict using K-Means
        prediction = kmeans.predict(scaled_input)[0]
        
        # Updated Mapping based on your model's output
        # According to your test: Cluster 1 is Setosa
        species_map = {
            1: "Setosa", 
            0: "Versicolor", 
            2: "Virginica"
        }
        
        current_species = species_map.get(prediction, "Unknown")
        
        # Show Results
        st.success(f"### Result: Cluster {prediction}")
        st.markdown(f"**Species Identification:** This flower matches the characteristics of **Iris-{current_species}**.")
    else:
        st.error("Error: Could not load 'scaler.pkl' or 'kmeans_model.pkl'. Please check if they are in your GitHub repository.")

