import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
from tempfile import NamedTemporaryFile

# Load the pre-trained deepfake detection model
try:
    model_path = r'C:\Users\loku0\OneDrive\Desktop\DeepFake\DeepFake_Model.h5'
    model = load_model(model_path)
except OSError:
    st.error("Error loading the model. Please check if the file exists and is not corrupted.")

# Preprocess each frame to fit the model input
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to the input size expected by the model
    frame_normalized = frame_resized / 255.0       # Normalize pixel values
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)  # Reshape to fit model input shape
    return frame_reshaped

# Predict if a frame is real (1) or fake (0)
def predict_frame(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0  # Assuming model outputs 1 for real, 0 for fake
    return predicted_class, prediction[0][0]  # Return the class and confidence

# Process video to classify frames based on FPS
def classify_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []

    # Calculate FPS to dynamically determine frame skipping
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Streamlit progress bar
    progress_bar = st.progress(0)

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()  # Read one frame
        if not ret:
            break

        frame_count += 1
        # Update progress bar
        progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Only process every 'frame_skip' frame
        if frame_count % frame_skip == 0:
            predicted_class, confidence = predict_frame(frame)
            results.append((predicted_class, confidence))

    cap.release()
    progress_bar.empty()

    # Count the majority result ('real' or 'fake')
    real_count = sum(1 for result in results if result[0] == 1)  # Count number of real frames
    fake_count = sum(1 for result in results if result[0] == 0)  # Count number of fake frames

    # Average confidence scores for real and fake
    avg_conf_real = np.mean([result[1] for result in results if result[0] == 1]) if real_count > 0 else 0
    avg_conf_fake = np.mean([result[1] for result in results if result[0] == 0]) if fake_count > 0 else 0

    # Determine the final result: majority voting
    video_result = 'real' if real_count > fake_count else 'fake'

    return video_result, real_count, fake_count, avg_conf_real, avg_conf_fake, total_frames

# Streamlit app
st.title('🎥 DeepFake Detection')
st.markdown("Welcome to the **DeepFake Detection **, where you can upload a video to determine whether it is real or fake.")

# Upload a video file
uploaded_file = st.file_uploader("📁 Upload your video file:", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name

    # Classify the uploaded video
    with st.spinner('🔍 Analyzing the video, please wait...'):
        video_result, real_count, fake_count, avg_conf_real, avg_conf_fake, total_frames = classify_video(temp_filename, frame_skip=10)

    # Display the results
    st.markdown(f"## 🏁 **Results**")
    if video_result == 'real':
        st.success(f"✅ The video is classified as **Real**.")
    else:
        st.error(f"❌ The video is classified as **Fake**.")

    st.write("### 📊 **Detailed Statistics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Frames Processed", total_frames)
        st.metric("Real Frames Count", real_count)
        st.metric("Fake Frames Count", fake_count)

    # Optionally, display the uploaded video
    st.markdown("### 🎬 **Uploaded Video**")
    st.video(uploaded_file)
