import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from keras.applications.resnet import preprocess_input
import os
import requests

# Page Configuration
st.set_page_config(
    page_title="Solar Panel Defect Classifier (Production)",
    page_icon="☀️",
    layout="wide"
)

CLASS_NAMES = [
    'Bird-drop', 
    'Clean', 
    'Dusty', 
    'Electrical-damage', 
    'Physical-Damage', 
    'Snow-Covered'
]

# Relative path configuration (No hardcoded Windows paths)
MODEL_PATH = 'models/best_model.keras'  

# ==========================================
# GOOGLE DRIVE CONFIGURATION
# ==========================================

GOOGLE_DRIVE_FILE_ID = st.secrets["GOOGLE_DRIVE_FILE_ID"]
# ==========================================

def download_model_from_drive(file_id, dest_path):
    """Downloads a large file from Google Drive bypassing the virus scan warning."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Send first request to get the warning cookie token
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    # If warning cookie exists, send request with confirmation token
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    # Save the file streamingly to prevent memory overflow
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_classifier_model():
    """Loads the trained model, downloading it from Google Drive if missing."""
    # 1. Check if model file exists locally
    if not os.path.exists(MODEL_PATH):
        if GOOGLE_DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
            st.error("Model file not found! Please configure your `GOOGLE_DRIVE_FILE_ID` in `app.py` so the app can download it on Streamlit Cloud.")
            return None
            
        try:
            with st.spinner("Downloading model weights from Google Drive (223 MB)... This may take a minute but only happens once."):
                download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error("Failed to download model weights from Google Drive. Please double check your File ID and check that the link sharing permissions are set to 'Anyone with the link'.")
            st.exception(e)
            return None

    try:
        # Load the model without compilation to avoid loading optimizer weights
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model from relative path '{MODEL_PATH}'. "
                 f"Please ensure the file is not corrupted and is an accessible `.keras` file.")
        st.exception(e)
        return None


def preprocess_image(image):
    """
    Resizes image to 224x224, converts to numpy array, adds batch dimension,
    and applies preprocess_input (ImageNet channel swap/mean subtraction) to match training.
    """
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    
    # Handle cases where image is grayscale (1 channel) or has alpha channel (4 channels)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    
    # Crucial: Apply preprocess_input to fix training/inference mismatch
    return preprocess_input(img_array)

def main():
    st.title("☀️ Solar Panel Defect Classification")
    st.markdown("Upload a close-up image of a solar panel (from drone or field inspection) to identify potential defects.")

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🖥️ System Status")
        st.markdown("✅ Model Ready  \n✅ Inspection Active")
        
        st.markdown("---")
        
        st.markdown("### Detectable Conditions")
        st.markdown("• Bird-Drop  \n• Dusty  \n• Electrical-damage  \n• Physical-Damage  \n• Clean")
        
        st.markdown("---")
        
        st.markdown("### Tech Stack")
        st.markdown("""
        **AI Model** - ResNet50, TensorFlow/Keras  
        **Processing** - ImageNet Normalization, Data Augmentation  
        **Frontend** - Streamlit
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Performance")
        st.markdown("""
        **Accuracy:** 86%  
        **F1 Score:** 87%
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Create two columns for clean layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            st.subheader("Inspection Results")
            model = load_classifier_model()
            
            if model:
                with st.spinner('Analyzing solar panel features...'):
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)
                    probs = predictions[0]

                # Create a DataFrame for visualization
                df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': probs})
                df = df.sort_values(by='Probability', ascending=False)

                # Display Top Predictions
                st.write("### Predictions Ranking")
                top_3 = df.head(3)
                
                # Check for critical defects in Top 1 prediction
                top_class = df.iloc[0]['Class']
                top_prob = df.iloc[0]['Probability']
                
                if top_class == 'Clean':
                    st.success(f"Status: **CLEAN** (Confidence: {top_prob*100:.2f}%)")
                else:
                    st.warning(f"Inspection Status: **{top_class.upper()}** (Confidence: {top_prob*100:.2f}%)")
                    st.info("💡 **Recommendation:** Trigger manual inspection or maintenance for this solar array subset.")
                
                st.markdown("---")
                
                for index, row in top_3.iterrows():
                    st.write(f"**{row['Class']}**")
                    st.progress(float(row['Probability']))
                    st.caption(f"{row['Probability']*100:.2f}% Confidence")

                st.divider()

                # Expandable all classes bar chart
                with st.expander("See complete probability distribution"):
                    st.bar_chart(df.set_index('Class'))
                    
                    # Formatting data table
                    formatted_df = df.copy()
                    formatted_df['Probability'] = formatted_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                    st.table(formatted_df)

if __name__ == "__main__":
    main()
