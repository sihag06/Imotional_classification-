# Speech Emotion Classification

A robust machine learning system that classifies emotions from speech audio using advanced feature extraction and deep neural networks.

##  Project Overview

This project implements an end-to-end speech emotion classification system capable of recognizing **7 emotional states** using a blend of traditional audio features and deep learning.

### Supported Emotions

- **Neutral** – Calm, emotionless speech  
- **Calm** – Peaceful, relaxed tone  
- **Happy** – Joyful, positive expressions  
- **Angry** – Aggressive, hostile speech  
- **Fearful** – Anxious, scared expressions  
- **Disgust** – Repulsed, disgusted tone  
- **Surprised** – Shocked, unexpected reactions  

> **Note:** The **Sad** emotion was excluded due to its significant overlap with Calm, which reduced model clarity. Removing it improved performance.

---

##  Features

- **Real-time Audio Processing:** Supports formats like WAV, MP3, FLAC, M4A  
- **Feature Extraction:** MFCC, Mel-spectrogram, Chroma, Spectral features  
- **Data Augmentation:** Time stretching, pitch shifting, noise injection  
- **Deep Neural Network:** BatchNorm, dropout, compact and efficient  
- **Web Interface:** Streamlit-based UI for easy interaction  
- **Performance:** Achieves ~95–97% training accuracy and ~80.28% test accuracy  

---

##  Model Training Details

### Dataset Handling

- **Class Balance:** Max 300 samples/class  
- **Train/Test Split:** 80/20 stratified  
- **Augmentation:** Applied 4× on training samples only  
- **Emotion Mapping:** Unified into 7 categories  

### Feature Engineering

- **Total Dimensions:** 181  
  - MFCCs (13 mean + 13 std + 13 delta) → 39  
  - Mel-spectrogram → 48  
  - Spectral (centroid, rolloff, ZCR) → 6  
  - Chroma → 12  

---

##  Model Architecture

```
Input Layer (181 features)
├── Dense (512) + ReLU + BatchNorm + Dropout(0.3)
├── Dense (256) + ReLU + BatchNorm + Dropout(0.3)
├── Dense (128) + ReLU + BatchNorm + Dropout(0.2)
├── Dense (64) + ReLU + Dropout(0.2)
└── Output Layer (7 classes) + Softmax
```

### Training Setup

- **Optimizer:** Adam (lr = 0.001)  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Batch Size:** 64  
- **Epochs:** Up to 100  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  

---

## 📊 Model Performance

| Metric        | Value     |
|---------------|-----------|
| **Train Accuracy** | ~95–97% |
| **Test Accuracy**  | 80.28%  |

### Classification Report (Test Data)

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Neutral   | 0.78      | 0.74   | 0.76     | 38      |
| Calm      | 0.84      | 0.87   | 0.85     | 60      |
| Happy     | 0.81      | 0.78   | 0.80     | 60      |
| Angry     | 0.81      | 0.85   | 0.83     | 60      |
| Fearful   | 0.77      | 0.78   | 0.78     | 60      |
| Disgust   | 0.87      | 0.69   | 0.77     | 39      |
| Surprised | 0.75      | 0.87   | 0.80     | 38      |

---

##  Technical Pipeline

### Preprocessing

- Resampling to 16kHz  
- Trimming silence  
- Normalizing duration to 3.5s  
- Feature extraction with scaling using `StandardScaler`

### Augmentation Techniques

- Time-stretch (0.9×)  
- Pitch-shift (+2 semitones)  
- Gaussian noise (0.5% amplitude)

---

##  Deployment

### Streamlit Web App

- Repository deployed on [Streamlit Cloud](https://streamlit.io/cloud)  
- Web interface allows real-time emotion prediction  
- Connected to GitHub and auto-deploys `app.py`

**Live App:**  
🔗 [https://emotion-classifier-vhm.streamlit.app/](https://emotion-classifier-vhm.streamlit.app/)

---
