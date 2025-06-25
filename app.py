import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import tempfile
import os

# Constants
FRAME_COUNT = 16
OUTPUT_FRAME_SIZE = (160, 160)

# Load the trained model
@st.cache_resource
def load_deepfake_model():
    return load_model("model/deepfake_detection_model_final.keras")

model = load_deepfake_model()
detector = MTCNN()

def extract_face(frame):
    results = detector.detect_faces(frame)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, OUTPUT_FRAME_SIZE)
    return cv2.resize(frame, OUTPUT_FRAME_SIZE)

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // FRAME_COUNT, 1)
    frames = []
    for i in range(FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extract_face(frame)
        frames.append(face)
    cap.release()
    while len(frames) < FRAME_COUNT:
        frames.append(np.zeros((*OUTPUT_FRAME_SIZE, 3), dtype=np.uint8))
    frames = np.array(frames) / 255.0
    return frames

st.title("🎬 Deepfake Video Detector")
st.write("Upload a short video (MP4) to check if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    # Save to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    st.video(uploaded_file)

    with st.spinner('Analyzing video...'):
        frames = extract_frames_from_video(tfile.name)
        frames_batch = np.expand_dims(frames, axis=0)  # (1, FRAME_COUNT, H, W, 3)
        pred = model.predict(frames_batch)
        fake_prob = float(pred[0, 1])
        real_prob = float(pred[0, 0])
        label = "Fake" if fake_prob > real_prob else "Real"
        confidence = max(fake_prob, real_prob)

    st.markdown(f"### Prediction: **:red[{label}]**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
    st.progress(confidence)
    os.unlink(tfile.name)
