import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time

# --- 1. PAGE CONFIG & MODERN UI STYLING ---
st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="centered")

# Custom CSS for a professional "Agri-Tech" look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2e7d32;
    }
    </style>
    """, unsafe_all_white_space=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load your exported files
    model = tf.keras.models.load_model("crop_disease_model.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    with open("treatment_info.json", "r") as f:
        treatments = json.load(f)
    return model, class_names, treatments

try:
    model, class_names, treatments = load_assets()
except Exception as e:
    st.error("Error loading model files. Ensure .h5 and .json files are in the same folder.")

# --- 3. UI HEADER ---
st.title("🌿 CropGuard AI")
st.markdown("### *AI-Powered Crop Doctor*")
st.write("Scan your crop leaves to detect diseases and get instant treatment plans.")

# --- 4. DUAL INPUT: CAMERA + UPLOAD ---
tab1, tab2 = st.tabs(["📸 Take a Photo", "📁 Upload Image"])

with tab1:
    cam_image = st.camera_input("Scan Leaf")

with tab2:
    uploaded_image = st.file_uploader("Choose an existing image...", type=["jpg", "jpeg", "png"])

# Logic to pick which image to use
input_image = cam_image if cam_image is not None else uploaded_image

# --- 5. PREDICTION LOGIC ---
if input_image is not None:
    # Display the image preview
    image = Image.open(input_image)
    st.image(image, caption='Captured Leaf', use_container_width=True)
    
    if st.button("🚀 Run Diagnosis"):
        with st.spinner('Analyzing symptoms...'):
            # Artificial delay for better UX
            time.sleep(1.5) 
            
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # AI Inference
            predictions = model.predict(img_array)
            idx = np.argmax(predictions[0])
            conf = np.max(predictions[0])
            label = class_names[idx]

            # --- 6. DISPLAY ATTRACTIVE RESULTS ---
            st.success("Analysis Complete!")
            
            # Result Card
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='color: #2e7d32; margin-top: 0;'>{label.replace('___', ' - ')}</h2>
                <p><b>Confidence Level:</b> {conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch treatment info
            info = treatments.get(label, treatments.get("Default"))
            
            st.divider()
            
            # Expander for treatment details
            col1, col2 = st.columns(2)
            with col1:
                st.warning(f"🧬 **Cause:** \n\n {info['cause']}")
                st.error(f"⚠️ **Severity:** \n\n {info.get('severity', 'Moderate')}")
            with col2:
                st.success(f"💊 **Treatment:** \n\n {info['treatment']}")
                st.info(f"🛡️ **Prevention:** \n\n {info['prevention']}")

# --- 7. FOOTER ---
st.markdown("---")
st.caption("Developed by a CSE Student at Osmania University | CropGuard v1.0")