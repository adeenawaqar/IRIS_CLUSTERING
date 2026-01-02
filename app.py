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
    except:
        return None, None, None

scaler, kmeans, dbscan = load_files()

# --- 2. UI Setup ---
st.set_page_config(page_title="Iris Clustering App")
st.title("🌸 Iris Flower Cluster Predictor")
st.write("Enter flower measurements to identify its cluster using K-Means.")

# --- 3. Sidebar: DBSCAN Analysis ---
st.sidebar.header("Model Comparison")
st.sidebar.write("Based on the original dataset analysis:")

# Instead of reading CSV, we show the results from your training
st.sidebar.info(f"""
- **K-Means Score:** 0.4590
- **DBSCAN Score:** 0.3492
""")

st.sidebar.warning("Note: K-Means is used for prediction because DBSCAN does not support the .predict() method for new data.")

# --- 4. Main Input Fields ---
col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length", min_value=0.0, value=5.1)
    sw = st.number_input("Sepal Width", min_value=0.0, value=3.5)

with col2:
    pl = st.number_input("Petal Length", min_value=0.0, value=1.4)
    pw = st.number_input("Petal Width", min_value=0.0, value=0.2)

# --- 5. Prediction Logic ---
if st.button("Identify Cluster"):
    if scaler is not None and kmeans is not None:
        # Create input array
        input_data = np.array([[sl, sw, pl, pw]])
        
        # Scale the data using saved scaler
        scaled_input = scaler.transform(input_data)
        
        # Predict using K-Means
        prediction = kmeans.predict(scaled_input)[0]
        
        # Show Results
        st.success(f"### Result: Cluster {prediction}")
        
        # Show Species Hint
        species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        st.write(f"This flower characteristics match **Iris-{species.get(prediction, 'Unknown')}**.")
    else:
        st.error("Error: Could not find 'scaler.pkl' or 'kmeans_model.pkl' on GitHub.")
