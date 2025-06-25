import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Load your trained model
@st.cache_resource
def load_deepfake_model():
    return load_model("model/deepfake_detection_model_final.keras")

model = load_deepfake_model()
detector = MTCNN()
FRAME_COUNT = 16
OUTPUT_FRAME_SIZE = (160, 160)

def extract_face(frame):
    results = detector.detect_faces(frame)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, OUTPUT_FRAME_SIZE)
    return cv2.resize(frame, OUTPUT_FRAME_SIZE)

def extract_frames_from_video(video_file):
    # video_file: BytesIO returned by st.file_uploader
    file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
    cap = cv2.VideoCapture()
    cap.open(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
    if not cap.isOpened():
        st.error("Could not open video file.")
        return np.zeros((FRAME_COUNT, *OUTPUT_FRAME_SIZE, 3), dtype=np.uint8)
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
    # If not enough frames, pad with zeros
    while len(frames) < FRAME_COUNT:
        frames.append(np.zeros((*OUTPUT_FRAME_SIZE, 3), dtype=np.uint8))
    return np.array(frames) / 255.0

st.title("Deepfake Video Detector")
st.write("Upload a short video file (e.g. mp4).")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    with st.spinner('Extracting faces and predicting...'):
        frames = extract_frames_from_video(uploaded_file)
        frames = np.expand_dims(frames, axis=0)  # (1, FRAME_COUNT, H, W, 3)
        pred = model.predict(frames)
        fake_prob = float(pred[0, 1])
        real_prob = float(pred[0, 0])
        label = "Fake" if fake_prob > real_prob else "Real"
        confidence = max(fake_prob, real_prob)
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
    st.progress(confidence)
