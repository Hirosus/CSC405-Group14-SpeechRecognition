import streamlit as st
import os
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import warnings

# Configuration
MODELS_DIR = "./models"
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
N_MFCC = 13

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HMM Voice AI", page_icon="🎙️")

@st.cache_resource
def load_models():
    """Loads trained HMM models and feature scaler."""
    models = {}
    scaler = None
    
    if not os.path.exists(MODELS_DIR):
        return None, None
    
    # Load HMM models
    for cmd in COMMANDS:
        path = os.path.join(MODELS_DIR, f"{cmd}.pkl")
        if os.path.exists(path):
            models[cmd] = joblib.load(path)
    
    # Load feature scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return models, scaler

def predict(models, scaler, audio_path):
    """Predicts the spoken command from an audio file."""
    # Load and preprocess audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Ensure consistent length of 1 second
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    else:
        y = y[:16000]
    
    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # Apply pre-emphasis filter
    y_pre = librosa.effects.preemphasis(y)
    
    # Extract MFCC features with deltas
    mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T
    
    # Apply feature normalization
    if scaler is not None:
        features = scaler.transform(features)
    
    # Score with all models
    best_score = float("-inf")
    best_label = "Unknown"
    scores = {}
    
    for cmd, model in models.items():
        try:
            score = model.score(features)
            scores[cmd] = score
            if score > best_score:
                best_score = score
                best_label = cmd
        except:
            scores[cmd] = -np.inf
            
    return best_label, scores, y, sr, features

# User interface
st.title("🎙️ HMM Voice Recognition")
st.markdown("### A Local-First Speech System")
st.write("Recognizes: " + ", ".join([c.upper() for c in COMMANDS]))

models, scaler = load_models()

if not models:
    st.error("❌ Models not found! Run 'train_model.py' first.")
else:
    st.success("✅ System Ready")
    
    # Input section
    st.divider()
    audio_val = st.audio_input("Record Voice Command")
    uploaded = st.file_uploader("Or Upload .WAV", type=["wav"])
    
    src = audio_val if audio_val else uploaded
    
    if src:
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(src.getvalue())
            path = tmp.name
        
        # Process audio and make prediction
        with st.spinner("Processing..."):
            label, scores, y, sr, feats = predict(models, scaler, path)
        
        st.metric(label="Prediction", value=label.upper())
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Waveform")
            fig, ax = plt.subplots(figsize=(4,2))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)
        with col2:
            st.caption("Spectrogram")
            fig2, ax2 = plt.subplots(figsize=(4,2))
            librosa.display.specshow(feats.T, sr=sr, ax=ax2)
            st.pyplot(fig2)
        
        # Clean up temporary file
        os.remove(path)