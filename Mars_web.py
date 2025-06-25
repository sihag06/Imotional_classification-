import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import tempfile
import os
from io import BytesIO


SAMPLING_RATE = 16000
N_MFCC = 13  
N_MELS = 48  
HOP_LENGTH = 512
N_FFT = 1024  
DURATION = 3.5


EMOTION_LABELS = ['Neutral', 'Calm', 'Happy', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def load_preprocess(file_path, sr=SAMPLING_RATE, duration=DURATION):
    try:
        x, _ = librosa.load(file_path, sr=sr, duration=duration, offset=0.2)
        x, _ = librosa.effects.trim(x, top_db=30)

        target_length = int(sr * duration)
        if len(x) < target_length:
            x = np.pad(x, (0, target_length - len(x)), mode='constant')
        else:
            x = x[:target_length]

        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val
        
        return x
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None

def extract_features_fast(x, sr):
    features = []
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    features.extend(np.mean(mfccs.T, axis=0))
    features.extend(np.std(mfccs.T, axis=0))    
    mfcc_delta = librosa.feature.delta(mfccs)
    features.extend(np.mean(mfcc_delta.T, axis=0))

    mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend(np.mean(mel_spec_db.T, axis=0))
    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr, hop_length=HOP_LENGTH)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, hop_length=HOP_LENGTH)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x, hop_length=HOP_LENGTH)[0]
    
    features.extend([
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
    ])

    chroma = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    features.extend(np.mean(chroma.T, axis=0))
    
    return np.array(features)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('best_model_corrected.h5')
        scaler = joblib.load('scaler_.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def predict_emotion(audio_file):
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return None, None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        audio_data = load_preprocess(tmp_file_path)
        if audio_data is None:
            return None, None
        
        features = extract_features_fast(audio_data, SAMPLING_RATE)
        features = features.reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        os.unlink(tmp_file_path)
        
        return predicted_class, confidence
        
    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        st.error(f"Error processing audio: {e}")
        return None, None

def main():
    st.set_page_config(
        page_title="Audio Emotion Classifier",
        page_icon="🎵",
        layout="centered"
    )
    
    st.title("🎵 Audio Emotion Classifier")
    st.write("Upload an audio file to classify the emotion expressed in speech")
    
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    st.subheader("🎯 Predicted Emotion")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing audio..."):
            predicted_class, confidence = predict_emotion(uploaded_file)
            
            if predicted_class is not None:
                emotion = EMOTION_LABELS[predicted_class]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Emotion", emotion)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                emotion_colors = {
                    'Neutral': '🟢',
                    'Calm': '🔵', 
                    'Happy': '🟡',
                    'Angry': '🔴',
                    'Fearful': '🟣',
                    'Disgust': '🟤',
                    'Surprised': '🟠'
                }
                
                st.success(f"{emotion_colors.get(emotion, '⚪')} The detected emotion is: **{emotion}**")
                
                st.progress(float(confidence))
                
            else:
                st.error("Failed to classify the audio. Please try with a different file.")
    
    else:
        st.info("👆 Please upload an audio file to get started")
    


if __name__ == "__main__":
    main()