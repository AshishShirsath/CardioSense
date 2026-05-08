import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tensorflow as tf
import pandas as pd
import pickle
import tempfile
import os

st.set_page_config(page_title="Heart Risk System", layout="centered")

st.title("🫀 Unified Heart Risk Prediction System")

# -------------------- SAFE MODEL LOADING --------------------
@st.cache_resource
def load_ecg_model():
    return tf.keras.models.load_model("best_model_1dc.keras")

@st.cache_resource
def load_rf_model():
    return pickle.load(open("random_forest_model.pkl", "rb"))

@st.cache_resource
def load_echo_model():
    device = torch.device("cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("edv_model_efficientnet.pth", map_location=device), strict=False)
    model.eval()
    return model, device

ecg_model = load_ecg_model()
rf_model = load_rf_model()
echo_model, device = load_echo_model()

labels = ['F', 'M', 'N', 'Q', 'S', 'V']

# -------------------- ECG PIPELINE --------------------
def preprocess_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise ValueError("Invalid ECG Image")
    img = cv2.resize(img, (512, 256))
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img

def binarize(img):
    _, thresh = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def clean_image(thresh):
    kernel = np.ones((2,2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def extract_signal(clean):
    signal = []
    for col in range(clean.shape[1]):
        rows = np.where(clean[:, col] == 255)[0]
        if len(rows) > 5:
            signal.append(np.median(rows))
        else:
            signal.append(np.nan)

    signal = np.array(signal)

    if np.all(np.isnan(signal)):
        return np.zeros(300)

    valid = ~np.isnan(signal)
    signal = signal[:np.where(valid)[0][-1] + 1]

    return signal

def postprocess(signal):
    signal = pd.Series(signal)
    signal = signal.interpolate().bfill().ffill()
    signal = -signal
    std = signal.std() if signal.std() != 0 else 1e-8
    return ((signal - signal.mean()) / std).values

def fix_length(signal, target_len=300):
    if len(signal) > target_len:
        return signal[:target_len]
    else:
        return np.pad(signal, (0, target_len - len(signal)))

def predict_ecg(path):
    img = preprocess_image(path)
    thresh = binarize(img)
    clean = clean_image(thresh)
    signal = extract_signal(clean)
    signal = postprocess(signal)
    signal = fix_length(signal)

    signal = signal.reshape(1, 300, 1)
    pred = ecg_model.predict(signal, verbose=0)
    cls = np.argmax(pred)

    return labels[cls], pred[0]

def ecg_score(label):
    return 0.1 if label == 'N' else 0.9 if label in ['V','S','F'] else 0.5

# -------------------- ECHO --------------------
def find_frames(video):
    cap = cv2.VideoCapture(video)
    intensities, frames = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensities.append(np.mean(gray))
        frames.append(frame)

    cap.release()
    intensities = np.array(intensities)

    return frames[np.argmin(intensities)], frames[np.argmax(intensities)]

def predict_volume(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        return echo_model(tensor).item()

def echo_score(edv, esv):
    ef = ((edv - esv) / edv) * 100

    if ef >= 55:
        return ef, 0.1
    elif ef >= 40:
        return ef, 0.5
    else:
        return ef, 0.9

# -------------------- UI --------------------

# -------- ECG --------
st.header("ECG Analysis")
ecg_file = st.file_uploader("Upload ECG Image", type=["png","jpg","jpeg"])
ecg_risk = 0

if ecg_file:
    path = "temp_ecg.png"
    with open(path, "wb") as f:
        f.write(ecg_file.read())

    try:
        label, probs = predict_ecg(path)
        ecg_risk = ecg_score(label)

        st.success(f"ECG Class: {label}")
        st.write(f"ECG Risk Score: {ecg_risk}")

    except Exception as e:
        st.error(e)
else:
    st.info("Upload ECG Image")

# -------- Echo --------
st.header("Echo Analysis")
video = st.file_uploader("Upload Echo Video", type=["avi"])
echo_risk_score = 0

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    ed_frame, es_frame = find_frames(tfile.name)

    edv = predict_volume(ed_frame)
    esv = predict_volume(es_frame)

    ef, echo_risk_score = echo_score(edv, esv)

    st.write(f"EDV: {edv:.2f}")
    st.write(f"ESV: {esv:.2f}")
    st.write(f"EF: {ef:.2f}%")

# -------- Clinical --------
st.header("Clinical Data")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("CP", [0,1,2,3])
bp = st.number_input("BP")
chol = st.number_input("Chol")
fbs = st.selectbox("FBS", [0,1])
restecg = st.selectbox("RestECG", [0,1,2])
thalach = st.number_input("Max HR")
exang = st.selectbox("Exang", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

clinical_score = 0

if st.button("Predict Clinical"):
    data = np.array([[age,sex,cp,bp,chol,fbs,restecg,
                      thalach,exang,oldpeak,slope,ca,thal]])

    clinical_score = rf_model.predict_proba(data)[0][1]
    st.write(f"Clinical Score: {clinical_score:.2f}")

# -------- FINAL --------
if st.button("Final Risk"):
    final = (0.4*clinical_score + 0.3*ecg_risk + 0.3*echo_risk_score)

    if final > 0.7:
        res = "🔴 HIGH RISK"
    elif final > 0.4:
        res = "🟠 MODERATE RISK"
    else:
        res = "🟢 LOW RISK"

    st.subheader("FINAL RESULT")
    st.metric("Score", f"{final:.2f}")
    st.write(res)