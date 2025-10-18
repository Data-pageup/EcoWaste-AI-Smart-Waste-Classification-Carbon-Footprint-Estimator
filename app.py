import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import json
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="EcoWaste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Hero Title */
    .hero-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #64ffda;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        border-color: rgba(16, 185, 129, 0.6);
    }
    
    .result-label {
        color: #64ffda;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        color: #10b981;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .result-sub {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Impact message */
    .impact-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .impact-title {
        color: #10b981;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .impact-text {
        color: #e2e8f0;
        font-size: 1.05rem;
        line-height: 1.7;
        text-align: center;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 2px dashed rgba(100, 255, 218, 0.4) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(100, 255, 218, 0.7) !important;
        background: rgba(30, 41, 59, 0.6) !important;
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        max-height: 400px !important;
    }
    
    [data-testid="stImage"] img {
        object-fit: cover !important;
        max-height: 400px !important;
    }
    
    /* Input */
    .stNumberInput > div > div > input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 0.7rem !important;
    }
    
    label {
        color: #64ffda !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Success alert */
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 12px !important;
        color: #10b981 !important;
    }
    
    /* Stats sidebar */
    .stats-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .stat-value {
        color: #10b981;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model paths
MODEL_DIR = Path("models")
ROOT_DIR = Path(".")

CLASSIFIER_PATH = MODEL_DIR / "waste_classifier_mobilenetv2_alpha035.keras" if (MODEL_DIR / "waste_classifier_mobilenetv2_alpha035.keras").exists() else ROOT_DIR / "waste_classifier_mobilenetv2_alpha035.keras"
REGRESSOR_PATH = MODEL_DIR / "co2_regressor_rfr.joblib" if (MODEL_DIR / "co2_regressor_rfr.joblib").exists() else ROOT_DIR / "co2_regressor_rfr.joblib"
COLUMNS_PATH = MODEL_DIR / "rfr_columns.json" if (MODEL_DIR / "rfr_columns.json").exists() else ROOT_DIR / "rfr_columns.json"
KMEANS_PATH = MODEL_DIR / "kmeans_users.joblib" if (MODEL_DIR / "kmeans_users.joblib").exists() else ROOT_DIR / "kmeans_users.joblib"
SCALER_PATH = MODEL_DIR / "users_scaler.joblib" if (MODEL_DIR / "users_scaler.joblib").exists() else ROOT_DIR / "users_scaler.joblib"

if 'user_history' not in st.session_state:
    st.session_state.user_history = []

@st.cache_resource
def load_models():
    models = {
        'classifier': None,
        'regressor': None,
        'rfr_columns': None,
        'kmeans': None,
        'scaler': None,
        'errors': []
    }
    
    try:
        if CLASSIFIER_PATH.exists():
            models['classifier'] = tf.keras.models.load_model(CLASSIFIER_PATH)
    except Exception as e:
        models['errors'].append(f"Classifier: {str(e)}")
    
    try:
        if REGRESSOR_PATH.exists():
            loaded_regressor = joblib.load(REGRESSOR_PATH)
            if isinstance(loaded_regressor, dict):
                for key in ['model', 'regressor', 'estimator']:
                    if key in loaded_regressor and hasattr(loaded_regressor[key], 'predict'):
                        models['regressor'] = loaded_regressor[key]
                        break
            else:
                models['regressor'] = loaded_regressor
    except Exception as e:
        models['errors'].append(f"Regressor: {str(e)}")
    
    try:
        if COLUMNS_PATH.exists():
            with open(COLUMNS_PATH, 'r') as f:
                models['rfr_columns'] = json.load(f)
    except Exception as e:
        models['errors'].append(f"Columns: {str(e)}")
    
    try:
        if KMEANS_PATH.exists():
            models['kmeans'] = joblib.load(KMEANS_PATH)
    except:
        pass
    
    try:
        if SCALER_PATH.exists():
            models['scaler'] = joblib.load(SCALER_PATH)
    except:
        pass
    
    return models

def preprocess_image(image, target_size=(160, 160)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def predict_waste_class(classifier, image):
    processed_img = preprocess_image(image)
    prediction = classifier.predict(processed_img, verbose=0)
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        waste_class = "R"
        confidence_pct = confidence * 100
    else:
        waste_class = "O"
        confidence_pct = (1 - confidence) * 100
    
    return waste_class, confidence_pct

def predict_co2_saved(regressor, rfr_columns, waste_class, weight_kg):
    """Fixed: Use material_O and material_R columns"""
    if isinstance(regressor, dict) or not hasattr(regressor, 'predict'):
        base_co2 = weight_kg * (0.5 if waste_class == "O" else 2.5)
        return base_co2
    
    # Initialize all features to 0
    features = {col: 0 for col in rfr_columns}
    
    # Set weight
    if 'weight_kg' in features:
        features['weight_kg'] = weight_kg
    
    # Set material type - FIXED to use material_O and material_R
    if waste_class == "O":
        if 'material_O' in features:
            features['material_O'] = 1
    else:  # waste_class == "R"
        if 'material_R' in features:
            features['material_R'] = 1
    
    # Convert to array
    feature_array = np.array([[features[col] for col in rfr_columns]])
    
    try:
        co2_saved = regressor.predict(feature_array)[0]
        return max(0, co2_saved)
    except Exception as e:
        # Fallback
        base_co2 = weight_kg * (0.5 if waste_class == "O" else 2.5)
        return base_co2

def main():
    # Header
    st.markdown('<h1 class="hero-title">♻️ ECOWASTE AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Waste Classification & Carbon Impact Analysis</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    if models['classifier'] is None or models['regressor'] is None:
        st.error("⚠️ Models not loaded. Check model files.")
        if models['errors']:
            for error in models['errors']:
                st.write(f"❌ {error}")
        st.stop()
    
    st.success("✅ System Ready")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Upload Waste Image")
        
        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp', 'jfif'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### ⚙️ Item Details")
        
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.01,
            max_value=100.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        classify_button = st.button("🚀 Analyze", use_container_width=True)
        
        # Stats
        if st.session_state.user_history:
            st.markdown("<br>", unsafe_allow_html=True)
            total_co2 = sum(item['co2'] for item in st.session_state.user_history)
            items = len(st.session_state.user_history)
            
            st.markdown(f"""
            <div class="stats-box">
                <div class="stat-value">{total_co2:.2f} kg</div>
                <div class="stat-label">Total CO₂ Saved</div>
            </div>
            <div class="stats-box">
                <div class="stat-value">{items}</div>
                <div class="stat-label">Items Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results
    if uploaded_file and classify_button:
        with st.spinner("Analyzing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress.progress(i + 1)
            
            waste_class, confidence = predict_waste_class(models['classifier'], image)
            co2_saved = predict_co2_saved(models['regressor'], models['rfr_columns'], waste_class, weight_kg)
            
            st.session_state.user_history.append({
                'class': waste_class,
                'weight': weight_kg,
                'co2': co2_saved
            })
            
            progress.empty()
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            class_name = "ORGANIC" if waste_class == "O" else "RECYCLABLE"
            icon = "🌱" if waste_class == "O" else "♻️"
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Classification</div>
                <div class="result-value">{icon}</div>
                <div class="result-sub">{class_name}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Confidence</div>
                <div class="result-value">{confidence:.1f}%</div>
                <div class="result-sub">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">CO₂ Saved</div>
                <div class="result-value">{co2_saved:.2f}</div>
                <div class="result-sub">kg CO₂e</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d:
            km_equivalent = co2_saved * 4.5
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Equivalent</div>
                <div class="result-value">{km_equivalent:.1f}</div>
                <div class="result-sub">km saved</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Impact message
        st.markdown(f"""
        <div class="impact-box">
            <div class="impact-title">💡 Environmental Impact</div>
            <div class="impact-text">
                By properly sorting this <strong>{class_name.lower()}</strong> waste, you've prevented 
                <strong>{co2_saved:.2f} kg of CO₂</strong> from entering the atmosphere—equivalent to 
                <strong>{km_equivalent:.1f} km</strong> of car emissions!<br><br>
                {'🌱 Composting prevents methane emissions from landfills.' if waste_class == 'O' else '♻️ Recycling saves energy and raw materials.'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64ffda; padding: 1.5rem;">
        <p style="font-size: 1.1rem; margin: 0;"><strong>ECOWASTE AI</strong> — Powered by TensorFlow</p>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">Every scan contributes to a sustainable future 🌍</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()