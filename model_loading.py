import requests

url = "https://github.com/Pranav-Uniyal/DEEPFAKE_DETECTION/releases/latest/download/DeepFake_Model.h5"

response = requests.get(url)
with open("DeepFake_Model.h5", "wb") as f:
    f.write(response.content)

print("Model downloaded successfully!")
