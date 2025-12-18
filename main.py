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
# Load Model (SAFE)
# -------------------------------
MODEL_PATH = "dfd-model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please ensure 'dfd-model.h5' is in the repository.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# Frame Preprocessing
# -------------------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# -------------------------------
# Frame Prediction
# -------------------------------
def predict_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed, verbose=0)
    confidence = float(prediction[0][0])
    predicted_class = 1 if confidence > 0.5 else 0
    return predicted_class, confidence

# -------------------------------
# Video Classification
# -------------------------------
def classify_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

    progress_bar = st.progress(0)
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

        if frame_count % frame_skip == 0:
            pred_class, confidence = predict_frame(frame)
            results.append((pred_class, confidence))

    cap.release()
    progress_bar.empty()

    if len(results) == 0:
        return "unknown", 0, 0, 0, 0, total_frames

    real_count = sum(1 for r in results if r[0] == 1)
    fake_count = sum(1 for r in results if r[0] == 0)

    avg_conf_real = np.mean([r[1] for r in results if r[0] == 1]) if real_count else 0
    avg_conf_fake = np.mean([r[1] for r in results if r[0] == 0]) if fake_count else 0

    video_result = "real" if real_count > fake_count else "fake"

    return video_result, real_count, fake_count, avg_conf_real, avg_conf_fake, total_frames

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎥 DeepFake Detection 🎭")
st.markdown(
    "Upload a video and this system will analyze frames using a deep learning model "
    "to determine whether the video is **Real** or **Fake**."
)

uploaded_file = st.file_uploader(
    "📁 Upload your video file",
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    with st.spinner("🔍 Analyzing video... Please wait"):
        (
            video_result,
            real_count,
            fake_count,
            avg_conf_real,
            avg_conf_fake,
            total_frames
        ) = classify_video(temp_video_path, frame_skip=10)

    st.markdown("## 🏁 Results")

    if video_result == "real":
        st.success("✅ The video is classified as **REAL**")
    elif video_result == "fake":
        st.error("❌ The video is classified as **FAKE**")
    else:
        st.warning("⚠ Unable to classify the video")

    st.markdown("### 📊 Frame Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Real Frames", real_count)
        st.metric("Fake Frames", fake_count)

    with col2:
        st.metric("Avg Real Confidence", f"{avg_conf_real:.2f}")
        st.metric("Avg Fake Confidence", f"{avg_conf_fake:.2f}")

    st.markdown("### 🎬 Uploaded Video")
    st.video(uploaded_file)


