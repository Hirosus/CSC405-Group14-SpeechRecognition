import os
import tarfile
import urllib.request
import numpy as np
import librosa
from hmmlearn import hmm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from glob import glob
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATA_DIR = "./dataset"
MODELS_DIR = "./models"
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# HMM hyperparameters
N_COMPONENTS = 5
N_ITER = 100

def download_and_extract_data():
    """Downloads and extracts the Google Speech Commands dataset if not present."""
    if not os.path.exists(DATA_DIR):
        print("Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, "speech_commands.tar.gz")
        with tarfile.open("speech_commands.tar.gz", "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print("Dataset ready")

def extract_features(audio_path):
    """
    Extracts MFCC features with delta and delta-delta coefficients.
    Returns a 39-dimensional feature vector per frame.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure consistent length of 1 second
        if len(y) < 16000: 
            y = np.pad(y, (0, 16000 - len(y)))
        else: 
            y = y[:16000]
        
        # Normalize amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Apply pre-emphasis filter
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, 
                                    hop_length=160, n_mels=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine features: 13 MFCCs + 13 deltas + 13 delta-deltas = 39 features
        features = np.vstack([mfcc, delta, delta2])
        
        return features.T
    except Exception as e:
        print(f"    Error processing file: {e}")
        return None

def train_hmm_models():
    """Trains HMM models for each command using normalized features."""
    if not os.path.exists(MODELS_DIR): 
        os.makedirs(MODELS_DIR)
    
    print("Starting HMM training pipeline")
    print("-" * 50)
    
    # Step 1: Extract features from all training files
    print("\nStep 1: Extracting features from training data")
    all_features = {cmd: [] for cmd in COMMANDS}
    all_features_combined = []
    
    for cmd in COMMANDS:
        print(f"  Processing '{cmd}'...", end=" ")
        audio_files = glob(os.path.join(DATA_DIR, cmd, "*.wav"))
        
        if len(audio_files) == 0:
            print("No files found")
            continue
        
        # Split into train and test sets
        train_files, _ = train_test_split(audio_files, train_size=0.8, random_state=42)
        train_files = train_files[:1500]
        
        # Extract features from each file
        for audio_file in train_files:
            features = extract_features(audio_file)
            if features is not None:
                all_features[cmd].append(features)
                all_features_combined.append(features)
        
        print(f"{len(all_features[cmd])} samples extracted")
    
    # Step 2: Fit feature scaler for normalization
    print("\nStep 2: Fitting feature scaler")
    combined_features = np.vstack(all_features_combined)
    print(f"  Total feature frames: {len(combined_features)}")
    
    scaler = StandardScaler()
    scaler.fit(combined_features)
    
    # Save scaler for inference
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")
    
    # Apply normalization to all features
    print("  Applying normalization to training features")
    for cmd in COMMANDS:
        all_features[cmd] = [scaler.transform(feat) for feat in all_features[cmd]]
    
    # Step 3: Train individual HMM for each command
    print("\nStep 3: Training HMM models")
    
    for cmd in COMMANDS:
        print(f"  Training model for '{cmd}'...", end=" ")
        
        if len(all_features[cmd]) == 0:
            print("No training data available")
            continue
        
        # Prepare training data
        feature_sequences = all_features[cmd]
        lengths = [len(seq) for seq in feature_sequences]
        X_train = np.vstack(feature_sequences)
        
        # Initialize and train Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type='diag',
            n_iter=N_ITER,
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train, lengths)
        
        # Save trained model
        model_path = os.path.join(MODELS_DIR, f"{cmd}.pkl")
        joblib.dump(model, model_path)
        print("Done")
    
    print("\n" + "-" * 50)
    print("Training complete")
    print(f"Models saved to: {MODELS_DIR}")
    print("Generated files:")
    print("  - 10 HMM models (yes.pkl, no.pkl, ...)")
    print("  - Feature scaler (scaler.pkl)")
    print("-" * 50)

if __name__ == "__main__":
    download_and_extract_data()
    train_hmm_models()