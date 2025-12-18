import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🎥",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "dfd-model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

model = load_model(MODEL_PATH)

# -------------------------------
# Face Detector
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Preprocess Frame (256x256)
# -------------------------------
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)

# -------------------------------
# Predict Single Frame (FACE-BASED)
# -------------------------------
def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    # Use largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = frame[y:y+h, x:x+w]

    processed = preprocess_frame(face)
    prediction = model.predict(processed, verbose=0)

    confidence = float(prediction[0][0])
    predicted_class = 1 if confidence > 0.5 else 0

    return predicted_class, confidence

# -------------------------------
# Temporal Smoothing
# -------------------------------
def temporal_smooth(results, window=5):
    smoothed = []
    for i in range(len(results)):
        window_slice = results[max(0, i-window):i+1]
        avg_conf = np.mean([r[1] for r in window_slice])
        cls = 1 if avg_conf > 0.5 else 0
        smoothed.append((cls, avg_conf))
    return smoothed

# -------------------------------
# Video Classification (SMART)
# -------------------------------
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

    # Analyze ~50 frames evenly
    sample_rate = max(total_frames // 50, 1)

    progress_bar = st.progress(0)
    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

        if frame_count % sample_rate == 0:
            pred, conf = predict_frame(frame)
            if pred is not None:
                results.append((pred, conf))

    cap.release()
    progress_bar.empty()

    # Not enough valid face frames
    if len(results) < 10:
        return "unknown", 0, 0, 0, 0, total_frames

    # Temporal smoothing
    results = temporal_smooth(results)

    # Confidence-weighted voting
    real_score = sum(conf for cls, conf in results if cls == 1)
    fake_score = sum(1 - conf for cls, conf in results if cls == 0)

    video_result = "real" if real_score > fake_score else "fake"

    real_count = sum(1 for r in results if r[0] == 1)
    fake_count = sum(1 for r in results if r[0] == 0)

    avg_conf_real = np.mean([r[1] for r in results if r[0] == 1]) if real_count else 0
    avg_conf_fake = np.mean([r[1] for r in results if r[0] == 0]) if fake_count else 0

    return video_result, real_count, fake_count, avg_conf_real, avg_conf_fake, total_frames

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎥 DeepFake Detection 🎭")
st.markdown(
    "This system analyzes **facial frames** from the video using a deep learning model "
    "and applies **temporal validation** for higher accuracy."
)

uploaded_file = st.file_uploader(
    "📁 Upload a video file",
    type=["mp4", "mov", "avi"]
)

if uploaded_file:
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    with st.spinner("🔍 Analyzing video..."):
        (
            video_result,
            real_count,
            fake_count,
            avg_conf_real,
            avg_conf_fake,
            total_frames
        ) = classify_video(temp_video_path)

    st.markdown("## 🏁 Results")

    if video_result == "real":
        st.success("✅ Video classified as **REAL**")
    elif video_result == "fake":
        st.error("❌ Video classified as **FAKE**")
    else:
        st.warning("⚠️ Unable to confidently classify this video")

    st.markdown("### 📊 Frame Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Real Frames", real_count)
        st.metric("Fake Frames", fake_count)

    with col2:
        st.metric("Avg Real Confidence", f"{avg_conf_real:.2f}")
        st.metric("Avg Fake Confidence", f"{avg_conf_fake:.2f}")

    st.markdown("### 🎬 Uploaded Video")
    st.video(uploaded_file)
