
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Custom CSS for a colorful look
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);}
    .stButton>button {background-color: #43c6ac; color: white;}
    .css-1d391kg {background: #fffbe7;}
    .css-1v0mbdj {color: #43c6ac;}
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Face Emotion Detection Dashboard")
st.markdown("Upload an image. The app will detect faces, classify emotions, and show a colorful dashboard of results.")

# Load face detector and emotion model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_colors = {
    'angry': '#e74c3c',
    'disgust': '#27ae60',
    'fear': '#8e44ad',
    'happy': '#f1c40f',
    'neutral': '#95a5a6',
    'sad': '#3498db',
    'surprise': '#e67e22'
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_counts = {label: 0 for label in emotion_labels}

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1).astype('float32') / 255.0
        prediction = model.predict(face)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = np.max(prediction)
        emotion_counts[emotion] += 1
        color = tuple(int(emotion_colors[emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_array, f'{emotion} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    st.image(img_array, caption='Detected Faces & Emotions', use_column_width=True)

    if len(faces) == 0:
        st.warning("No faces detected.")
    else:
        # Dashboard: Bar graph of emotions
        st.subheader("Emotion Distribution")
        fig, ax = plt.subplots()
        bars = ax.bar(emotion_labels, [emotion_counts[e] for e in emotion_labels],
                      color=[emotion_colors[e] for e in emotion_labels])
        ax.set_ylabel("Count")
        ax.set_xlabel("Emotion")
        ax.set_title("Detected Emotions")
        for bar, count in zip(bars, [emotion_counts[e] for e in emotion_labels]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                    ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)
else:
    st.info("Please upload an image.")

st.markdown("""
---
**Tips:**  
- The model file must be named `emotion_model.h5` and be in the same folder as this script.  
- You can train your own model or download one from [Hugging Face](https://huggingface.co/models) (search for 'emotion detection').  
""")