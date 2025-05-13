
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the model
model = load_model("emotion_model_compile.keras")  # Or use .h5 depending on what you prefer

# Emotion categories - update these to match your model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Real-time Emotion Detection")

frame = st.camera_input("Take a picture")

if frame is not None:
    img = Image.open(frame)
    img = np.array(img.convert('RGB'))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            prediction = model.predict(face)
            emotion_idx = np.argmax(prediction)
            emotion_text = emotion_labels[emotion_idx]

            st.success(f"Detected Emotion: **{emotion_text}**")

            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        st.image(img, caption="Detected Face(s)")
