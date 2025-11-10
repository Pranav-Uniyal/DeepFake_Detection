
# 🎭 DeepFake Detection System

A **Streamlit-based web application** to detect deepfake videos using a CNN model. This tool analyzes frames of a video and determines whether the video is **real** or **fake** using majority voting and confidence scoring.

---
<img width="511" height="240" alt="image" src="https://github.com/user-attachments/assets/2760b98e-ac25-44a9-be3d-08c2dde64d25" />

# [Kaggle Notebook Link](https://www.kaggle.com/code/fall2fire/deepfake-detection-96-accuracy)
---
## 📽️ Demo
![image](https://github.com/user-attachments/assets/3c15ea39-3e8d-41fd-82b8-ee6fb8176517)
![image](https://github.com/user-attachments/assets/15de49f5-646d-4946-b8fa-731190c80302)



Try out the application by uploading a `.mp4`, `.mov`, or `.avi` file and let the model classify its authenticity!

---

## 🚀 Features

- 📤 Upload and analyze video files in real-time.
- 🧠 Deep learning-based frame-level prediction using a CNN model.
- 📊 Visual feedback on number of real vs fake frames.
- 📈 Confidence scores for both classes.
- 🧾 Interactive and intuitive Streamlit interface.

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

## 📦 Requirements

Install the required dependencies before running the app:

```bash
pip install streamlit opencv-python-headless numpy tensorflow
```

> ⚠️ You also need to have a trained model file named `DeepFake_Model.h5`. Update the `model_path` variable in the script to point to the correct location.

---

## 🧑‍💻 Run the App

1. **Clone the repository**:

```bash
git clone https://github.com/Pranav-Uniyal/DeepFake_Detection.git
cd DeepFake_Detection
```

2. **Place your model**:

Place your `DeepFake_Model.h5` in the project root or update the `model_path` accordingly in the script.

3. **Run the Streamlit app**:

```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
Deepfake-Detection(Repo)
│
├── main.py  # Main Streamlit app
├── dfd-model.h5                       # Pre-trained CNN model
└── README.md                               # Project documentation
```

---
## Download the Model
Releases-[Download](https://github.com/Pranav-Uniyal/DeepFake_Detection/releases/tag/model)

---

## 📊 Output

After uploading a video, the app displays:

- ✅ Whether the video is **real** or **fake**
- 📈 Number of real and fake frames
- 🎬 Embedded video preview

---

## ❓ How it Works

- Video is read frame by frame using OpenCV.
- Every N-th frame (default: 10) is passed through a CNN model.
- Results are collected and majority class (real/fake) is used as final verdict.
- Confidence scores are averaged per class for deeper insight.

---

## 🧪 Example

| Frame # | Prediction | Confidence |
|---------|------------|------------|
| 10      | Fake       | 0.23       |
| 20      | Real       | 0.86       |
| ...     | ...        | ...        |

---

## 📌 Notes

- Ensure your model accepts `(224, 224, 3)` input shape.
- The app assumes binary classification: **1 = Real**, **0 = Fake**.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- TensorFlow/Keras
- Streamlit
- OpenCV
