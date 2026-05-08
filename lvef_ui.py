import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(page_title="EchoNet EDV/ESV Predictor", layout="centered")

st.title("🫀 Echocardiogram Analysis (EDV + ESV + EF)")
st.write("Upload Echo video → Predict EDV, ESV → Compute EF")

# -------------------- MODEL --------------------
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# -------------------- FRAME DETECTION --------------------
def find_ed_es_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    intensities = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensities.append(np.mean(gray))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Video could not be processed")

    intensities = np.array(intensities)

    # ✅ FIXED (more reliable assumption)
    ed_idx = np.argmax(intensities)   # ED → larger volume
    es_idx = np.argmin(intensities)   # ES → smaller volume

    return frames[ed_idx], frames[es_idx], ed_idx, es_idx, intensities

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNRegressor().to(device)
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    new_state_dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("model."):
            new_state_dict["model." + k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model, device

# -------------------- PREDICT --------------------
def predict_volume(frame, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        return model(tensor).item()

# -------------------- UI --------------------
uploaded_video = st.file_uploader("Upload Echo Video (.avi)", type=["avi"])

edv_model_path = "edv_model_efficientnet.pth"
esv_model_path = "esv_model_efficientnet.pth"

if uploaded_video:
    try:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.success("Video Uploaded Successfully")

        # Frame detection
        ed_frame, es_frame, ed_idx, es_idx, intensities = find_ed_es_frames(tfile.name)

        st.subheader("Frame Detection")
        st.write(f"ED Frame Index: {ed_idx}")
        st.write(f"ES Frame Index: {es_idx}")

        # Plot intensity
        fig, ax = plt.subplots()
        ax.plot(intensities)
        ax.scatter(ed_idx, intensities[ed_idx], color="red", label="ED")
        ax.scatter(es_idx, intensities[es_idx], color="green", label="ES")
        ax.legend()
        st.pyplot(fig)

        # Show frames
        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2.cvtColor(ed_frame, cv2.COLOR_BGR2RGB), caption="ED Frame")

        with col2:
            st.image(cv2.cvtColor(es_frame, cv2.COLOR_BGR2RGB), caption="ES Frame")

        # Load models
        edv_model, device = load_model(edv_model_path)
        esv_model, _ = load_model(esv_model_path)

        # Predict
        edv = predict_volume(ed_frame, edv_model, device)
        esv = predict_volume(es_frame, esv_model, device)

        # ---------------- DEBUG ----------------
        st.write("Raw EDV:", edv)
        st.write("Raw ESV:", esv)

        # ---------------- SCALING FIX ----------------
        # 👉 adjust this based on training
        if edv < 5 and esv < 5:
            st.warning("Scaling outputs (model likely normalized)")
            edv *= 200
            esv *= 200

        # ---------------- SWAP FIX ----------------
        if esv > edv:
            st.warning("Swapping EDV & ESV (frame mismatch correction)")
            edv, esv = esv, edv

        # ---------------- EF ----------------
        if edv > 0:
            ef = ((edv - esv) / edv) * 100
        else:
            ef = 0

        # Display
        st.subheader("Cardiac Volumes")

        c1, c2, c3 = st.columns(3)
        c1.metric("EDV", f"{edv:.2f} mL")
        c2.metric("ESV", f"{esv:.2f} mL")
        c3.metric("EF", f"{ef:.2f} %")

        # Interpretation
        st.subheader("Assessment")

        if ef >= 55:
            st.success("🟢 Normal")
        elif ef >= 40:
            st.warning("🟠 Moderate Dysfunction")
        else:
            st.error("🔴 High Risk (Low EF)")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a video to begin")