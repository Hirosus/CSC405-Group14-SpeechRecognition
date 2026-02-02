import os
import numpy as np
import joblib
import librosa
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "./dataset"
MODELS_DIR = "./models"
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

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
        
        # Combine features
        features = np.vstack([mfcc, delta, delta2])
        
        return features.T
    except Exception as e:
        return None

def evaluate_models():
    """Evaluates trained HMM models on the test set."""
    print("Model Evaluation")
    print("-" * 50)
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print("Error: Models directory not found")
        print("Please run train_model.py first")
        return
    
    # Load trained models
    print("\nLoading models...")
    models = {}
    for cmd in COMMANDS:
        model_path = os.path.join(MODELS_DIR, f"{cmd}.pkl")
        if os.path.exists(model_path):
            models[cmd] = joblib.load(model_path)
            print(f"  Loaded {cmd}.pkl")
        else:
            print(f"  Warning: {cmd}.pkl not found")
    
    # Load feature scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  Loaded scaler.pkl")
    else:
        print("  Warning: scaler.pkl not found")
        print("  Results may be inaccurate without proper normalization")
        scaler = None
    
    if len(models) != len(COMMANDS):
        print(f"\nError: Expected {len(COMMANDS)} models, found {len(models)}")
        return
    
    # Evaluate on test set
    y_true = []
    y_pred = []
    
    print("\n" + "-" * 50)
    print("Testing on held-out test set")
    print("-" * 50)
    
    for cmd in COMMANDS:
        print(f"  Testing '{cmd}'...", end=" ")
        
        audio_files = glob(os.path.join(DATA_DIR, cmd, "*.wav"))
        if len(audio_files) == 0:
            print("No test files found")
            continue
        
        # Use same train/test split as training
        _, test_files = train_test_split(audio_files, train_size=0.8, random_state=42)
        
        # Evaluate on first 100 test samples
        for audio_file in test_files[:100]:
            features = extract_features(audio_file)
            
            if features is None:
                continue
            
            # Apply feature normalization
            if scaler is not None:
                features = scaler.transform(features)
            
            # Score with all models
            best_score = float("-inf")
            best_label = None
            
            for label, model in models.items():
                try:
                    score = model.score(features)
                    if score > best_score:
                        best_score = score
                        best_label = label
                except:
                    pass
            
            if best_label is not None:
                y_true.append(cmd)
                y_pred.append(best_label)
        
        print(f"{len([y for y in y_true if y == cmd])} samples tested")
    
    # Check if evaluation succeeded
    if len(y_pred) == 0:
        print("\nError: No predictions were made")
        print("Please check that models were trained correctly")
        return
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "-" * 50)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Total samples evaluated: {len(y_pred)}")
    print("-" * 50)
    
    # Generate classification report
    print("\nClassification Report")
    print("-" * 50)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=COMMANDS,
        labels=COMMANDS,
        zero_division=0
    )
    print(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=COMMANDS)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=COMMANDS, 
        yticklabels=COMMANDS, 
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Display per-class accuracy
    print("\n" + "-" * 50)
    print("Per-Class Accuracy")
    print("-" * 50)
    for i, cmd in enumerate(COMMANDS):
        correct = cm[i, i]
        total = cm[i, :].sum()
        if total > 0:
            class_accuracy = (correct / total) * 100
            print(f"  {cmd:>6}: {class_accuracy:5.1f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_models()