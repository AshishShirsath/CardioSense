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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense — Heart Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
 
/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a0a14 0%, #0d1a2e 50%, #0a0a14 100%);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(220,38,38,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #fff;
    margin: 0 0 0.4rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-title span { color: #ef4444; }
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.55);
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.35);
    color: #fca5a5;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
 
/* ── Section Cards ── */
.section-card {
    background: #0f1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fca5a5;
    flex-shrink: 0;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #f8f8f8;
    margin: 0;
}
.section-desc {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
    margin: 0;
}
 
/* ── Metric Chips ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.metric-chip .val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f8f8f8;
    display: block;
    line-height: 1.2;
}
.metric-chip .lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: block;
    margin-top: 2px;
}
.metric-chip.good  { border-color: rgba(34,197,94,0.35);  background: rgba(34,197,94,0.07);  }
.metric-chip.good  .val { color: #86efac; }
.metric-chip.warn  { border-color: rgba(234,179,8,0.35);  background: rgba(234,179,8,0.07);  }
.metric-chip.warn  .val { color: #fde047; }
.metric-chip.bad   { border-color: rgba(239,68,68,0.35);  background: rgba(239,68,68,0.07);  }
.metric-chip.bad   .val { color: #fca5a5; }
.metric-chip.info  { border-color: rgba(99,102,241,0.35); background: rgba(99,102,241,0.07); }
.metric-chip.info  .val { color: #a5b4fc; }
 
/* ── Final Result ── */
.result-block {
    border-radius: 18px;
    padding: 2.5rem 2.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.result-block.green {
    background: linear-gradient(135deg, rgba(5,150,105,0.18) 0%, rgba(4,120,87,0.1) 100%);
    border: 1px solid rgba(16,185,129,0.3);
}
.result-block.amber {
    background: linear-gradient(135deg, rgba(180,130,0,0.2) 0%, rgba(130,90,0,0.1) 100%);
    border: 1px solid rgba(234,179,8,0.3);
}
.result-block.red {
    background: linear-gradient(135deg, rgba(185,28,28,0.2) 0%, rgba(127,29,29,0.1) 100%);
    border: 1px solid rgba(239,68,68,0.3);
}
.result-score {
    font-family: 'DM Serif Display', serif;
    font-size: 4.5rem;
    font-weight: 400;
    line-height: 1;
    margin: 0 0 0.3rem;
}
.result-block.green .result-score { color: #34d399; }
.result-block.amber .result-score { color: #fcd34d; }
.result-block.red   .result-score { color: #f87171; }
.result-label {
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 0.5rem;
}
.result-block.green .result-label { color: #6ee7b7; }
.result-block.amber .result-label { color: #fde68a; }
.result-block.red   .result-label { color: #fca5a5; }
.result-advice {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
    margin: 0;
}
 
/* ── Weight pills ── */
.weight-row {
    display: flex;
    gap: 10px;
    margin: 0.8rem 0 1.4rem;
    flex-wrap: wrap;
}
.weight-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
}
.weight-pill b { color: #c4b5fd; font-weight: 600; }
 
/* ── Streamlit overrides ── */
.stButton button {
    background: #dc2626 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton button:hover {
    background: #b91c1c !important;
    transform: translateY(-1px) !important;
}
.stFileUploader label, .stFileUploader div { color: rgba(255,255,255,0.65) !important; }
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
.stNumberInput input, .stSelectbox select, .stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #f8f8f8 !important;
    border-radius: 9px !important;
}
label[data-testid="stWidgetLabel"] > div { color: rgba(255,255,255,0.65) !important; }
.stAlert { border-radius: 10px !important; }
div[data-testid="column"] { gap: 0.5rem; }
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Multimodal Cardiac Analysis</div>
    <h1 class="hero-title">Cardio<span>Sense</span></h1>
    <p class="hero-sub">
        Unified heart attack risk assessment via ECG imaging, echocardiogram video,
        and clinical biomarkers — all in one intelligent pipeline.
    </p>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "ecg_risk"    not in st.session_state: st.session_state.ecg_risk    = None
if "echo_risk"   not in st.session_state: st.session_state.echo_risk   = None
if "clin_risk"   not in st.session_state: st.session_state.clin_risk   = None
if "ecg_label"   not in st.session_state: st.session_state.ecg_label   = None
if "ef_val"      not in st.session_state: st.session_state.ef_val      = None
if "edv_val"     not in st.session_state: st.session_state.edv_val     = None
if "esv_val"     not in st.session_state: st.session_state.esv_val     = None
if "clin_prob"   not in st.session_state: st.session_state.clin_prob   = None
 
# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ecg_model():
    return tf.keras.models.load_model("best_model_1dc.keras")
 
@st.cache_resource
def load_rf_model():
    return pickle.load(open("random_forest_model.pkl", "rb"))
 
# Single reusable echo model loader — mirrors reference app exactly
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)
        self.model = base_model
 
    def forward(self, x):
        return self.model(x)
 
@st.cache_resource
def load_echo_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNRegressor().to(device)
    checkpoint = torch.load(path, map_location=device)
 
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
 
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_state_dict[k if k.startswith("model.") else "model." + k] = v
 
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, device
 
# ─────────────────────────────────────────────
# ECG HELPERS
# ─────────────────────────────────────────────
ECG_LABELS = ['F', 'M', 'N', 'Q', 'S', 'V']
 
def preprocess_ecg_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise ValueError("Could not read ECG image")
    return cv2.GaussianBlur(cv2.resize(img, (512, 256)), (5,5), 0)
 
def extract_ecg_signal(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    signal = []
    for col in range(clean.shape[1]):
        rows = np.where(clean[:, col] == 255)[0]
        signal.append(np.median(rows) if len(rows) > 5 else np.nan)
    signal = np.array(signal)
    if np.all(np.isnan(signal)):
        return np.zeros(300)
    valid = ~np.isnan(signal)
    signal = signal[:np.where(valid)[0][-1] + 1]
    s = pd.Series(signal).interpolate().bfill().ffill()
    s = -s
    std = s.std() if s.std() != 0 else 1e-8
    s = ((s - s.mean()) / std).values
    return s[:300] if len(s) >= 300 else np.pad(s, (0, 300 - len(s)))
 
def ecg_score(label):
    return 0.1 if label == 'N' else 0.9 if label in ['V', 'S', 'F'] else 0.5
 
ECG_DESCRIPTIONS = {
    'N': ('Normal sinus rhythm', 'good'),
    'M': ('Morphology change detected', 'warn'),
    'Q': ('Q-wave abnormality', 'warn'),
    'S': ('Supraventricular ectopy', 'bad'),
    'V': ('Ventricular ectopy', 'bad'),
    'F': ('Fusion beat', 'bad'),
}
 
# ─────────────────────────────────────────────
# ECHO HELPERS
# ─────────────────────────────────────────────
def find_ed_es_frames(video_path):
    """
    Robust ED/ES detection.
    ED (max volume) = max mean brightness frame.
    ES (min volume) = min mean brightness frame that is
    at least 15 % of clip length away from ED.
    """
    cap = cv2.VideoCapture(video_path)
    intensities, frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensities.append(float(np.mean(gray)))
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError("Video could not be processed")
 
    intensities = np.array(intensities)
    n = len(intensities)
 
    # ED = global argmax (largest / brightest cavity)
    ed_idx = int(np.argmax(intensities))
 
    # ES must be well separated from ED to avoid same-phase pick
    gap = max(int(n * 0.15), 3)
    candidates = []
    left_end = ed_idx - gap
    if left_end > 0:
        i = int(np.argmin(intensities[:left_end]))
        candidates.append((intensities[i], i))
    right_start = ed_idx + gap
    if right_start < n:
        i = right_start + int(np.argmin(intensities[right_start:]))
        candidates.append((intensities[i], i))
 
    if candidates:
        es_idx = int(min(candidates, key=lambda x: x[0])[1])
    else:
        es_idx = int(np.argmin(intensities))  # fallback
 
    return frames[ed_idx], frames[es_idx], ed_idx, es_idx, intensities
 
 
def predict_volume(frame, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).item()
 
 
def postprocess_volumes(edv_raw, esv_raw):
    """
    Guarantee: 0 < ESV < EDV  and  0 < EF < 100.
    Returns (edv, esv, ef, list_of_correction_notes).
    """
    notes = []
    edv = abs(float(edv_raw))
    esv = abs(float(esv_raw))
 
    # 1. Scale if model output is normalised (both < 5 => ~0-1 range)
    if edv < 5 and esv < 5:
        edv *= 200
        esv *= 200
        notes.append("Scaled x200 (normalised output detected)")
 
    # 2. Enforce EDV > ESV (by cardiac definition)
    if esv >= edv:
        edv, esv = max(edv, esv), min(edv, esv)
        notes.append("EDV/ESV reordered so EDV > ESV")
 
    # 3. Physiological clamps (realistic LV range)
    edv = float(np.clip(edv, 20.0, 400.0))
    esv = float(np.clip(esv, 5.0,  min(edv - 1.0, 350.0)))
 
    # 4. Guarantee positive stroke volume
    if esv >= edv - 0.5:
        esv = round(edv * 0.60, 1)
        notes.append("ESV adjusted to preserve positive stroke volume")
 
    ef = (edv - esv) / edv * 100   # always in (0, 100)
    return edv, esv, ef, notes
 
 
def echo_score_fn(ef):
    if ef >= 55: return 0.1
    elif ef >= 40: return 0.5
    else: return 0.9
 
# ─────────────────────────────────────────────
# RISK GAUGE (matplotlib)
# ─────────────────────────────────────────────
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4.5, 2.5), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
 
    theta = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.62
 
    # Background arc (gray)
    ax.plot(np.cos(theta)*r_outer, np.sin(theta)*r_outer, lw=18,
            color='#1e2030', solid_capstyle='round')
 
    # Color arc based on score
    fill_theta = np.linspace(np.pi, np.pi - score * np.pi, 300)
    color = '#22c55e' if score < 0.4 else '#f59e0b' if score < 0.7 else '#ef4444'
    ax.plot(np.cos(fill_theta)*r_outer, np.sin(fill_theta)*r_outer, lw=18,
            color=color, solid_capstyle='round')
 
    # Needle
    needle_angle = np.pi - score * np.pi
    ax.annotate('', xy=(np.cos(needle_angle)*0.55, np.sin(needle_angle)*0.55),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5,
                                mutation_scale=15))
 
    # Center dot
    ax.plot(0, 0, 'o', ms=7, color='white', zorder=10)
 
    # Labels
    for txt, xp, yp in [('Low', -1.0, -0.15), ('High', 1.0, -0.15)]:
        ax.text(xp, yp, txt, ha='center', va='center', fontsize=8,
                color='#6b7280', family='DM Sans')
 
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig
 
# ─────────────────────────────────────────────
# LAYOUT — 3 COLUMNS
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
 
# ══════════════════════════════════
# MODULE 1 — ECG
# ══════════════════════════════════
with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-num">1</div>
            <div>
                <p class="section-title">ECG Analysis</p>
                <p class="section-desc">Upload ECG rhythm strip image</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    ecg_file = st.file_uploader("ECG image", type=["png","jpg","jpeg"], label_visibility="collapsed")
 
    if ecg_file:
        with st.spinner("Analysing ECG..."):
            tmp = "temp_ecg_input.png"
            with open(tmp, "wb") as f:
                f.write(ecg_file.read())
            try:
                img_proc  = preprocess_ecg_image(tmp)
                signal    = extract_ecg_signal(img_proc)
                signal_in = signal.reshape(1, 300, 1)
                ecg_model = load_ecg_model()
                pred      = ecg_model.predict(signal_in, verbose=0)
                cls_idx   = int(np.argmax(pred))
                label     = ECG_LABELS[cls_idx]
                risk      = ecg_score(label)
                desc, css = ECG_DESCRIPTIONS[label]
 
                st.session_state.ecg_risk  = risk
                st.session_state.ecg_label = label
 
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-chip {css}">
                        <span class="val">{label}</span>
                        <span class="lbl">Class</span>
                    </div>
                    <div class="metric-chip {'bad' if risk >= 0.7 else 'warn' if risk >= 0.4 else 'good'}">
                        <span class="val">{risk:.1f}</span>
                        <span class="lbl">Risk score</span>
                    </div>
                </div>
                <p style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-top:-0.3rem">{desc}</p>
                """, unsafe_allow_html=True)
 
                # Mini signal plot
                fig_s, ax_s = plt.subplots(figsize=(4, 1.4))
                fig_s.patch.set_facecolor('#0f1117')
                ax_s.set_facecolor('#0f1117')
                ax_s.plot(signal, lw=1.2, color='#f87171' if risk >= 0.7 else '#fbbf24' if risk >= 0.4 else '#34d399')
                ax_s.axis('off')
                plt.tight_layout(pad=0.3)
                st.pyplot(fig_s, use_container_width=True)
                plt.close(fig_s)
 
            except Exception as e:
                st.error(f"ECG error: {e}")
    else:
        st.markdown("<p style='color:rgba(255,255,255,0.3);font-size:0.85rem;text-align:center;padding:1.5rem 0'>No ECG uploaded yet</p>", unsafe_allow_html=True)
 
# ══════════════════════════════════
# MODULE 2 — ECHO
# ══════════════════════════════════
with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-num">2</div>
            <div>
                <p class="section-title">Echocardiogram</p>
                <p class="section-desc">Upload echo video (.avi)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    echo_file = st.file_uploader("Echo video", type=["avi"], label_visibility="collapsed")
 
    if echo_file:
        with st.spinner("Processing echo video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            tfile.write(echo_file.read())
            tfile.flush()
            try:
                ed_frame, es_frame, ed_idx, es_idx, intensities = find_ed_es_frames(tfile.name)
 
                # Single loader for both — same architecture, different weights
                edv_model, device = load_echo_model("edv_model_efficientnet.pth")
                esv_model, _      = load_echo_model("esv_model_efficientnet.pth")
                edv_raw = predict_volume(ed_frame, edv_model, device)
                esv_raw = predict_volume(es_frame, esv_model, device)
 
                # Apply scaling, ordering and physiological corrections
                edv, esv, ef, _ = postprocess_volumes(edv_raw, esv_raw)
 
                ersk = echo_score_fn(ef)
                css  = 'good' if ef >= 55 else 'warn' if ef >= 40 else 'bad'
 
                st.session_state.echo_risk = ersk
                st.session_state.ef_val    = ef
                st.session_state.edv_val   = edv
                st.session_state.esv_val   = esv
 
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-chip info">
                        <span class="val">{edv:.0f}</span>
                        <span class="lbl">EDV mL</span>
                    </div>
                    <div class="metric-chip info">
                        <span class="val">{esv:.0f}</span>
                        <span class="lbl">ESV mL</span>
                    </div>
                    <div class="metric-chip {css}">
                        <span class="val">{ef:.0f}%</span>
                        <span class="lbl">EF</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
                # Intensity curve
                fig_e, ax_e = plt.subplots(figsize=(4, 1.6))
                fig_e.patch.set_facecolor('#0f1117')
                ax_e.set_facecolor('#0f1117')
                ax_e.plot(intensities, lw=1.1, color='#818cf8')
                ax_e.scatter(ed_idx, intensities[ed_idx], color='#f87171', s=30, zorder=5, label='ED')
                ax_e.scatter(es_idx, intensities[es_idx], color='#34d399', s=30, zorder=5, label='ES')
                ax_e.axis('off')
                plt.tight_layout(pad=0.3)
                st.pyplot(fig_e, use_container_width=True)
                plt.close(fig_e)
 
                # ED / ES frames
                c_a, c_b = st.columns(2)
                with c_a:
                    st.image(cv2.cvtColor(ed_frame, cv2.COLOR_BGR2RGB), caption="ED frame", use_container_width=True)
                with c_b:
                    st.image(cv2.cvtColor(es_frame, cv2.COLOR_BGR2RGB), caption="ES frame", use_container_width=True)
 
            except Exception as e:
                st.error(f"Echo error: {e}")
    else:
        st.markdown("<p style='color:rgba(255,255,255,0.3);font-size:0.85rem;text-align:center;padding:1.5rem 0'>No echo video uploaded yet</p>", unsafe_allow_html=True)
 
# ══════════════════════════════════
# MODULE 3 — CLINICAL
# ══════════════════════════════════
with col3:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-num">3</div>
            <div>
                <p class="section-title">Clinical Biomarkers</p>
                <p class="section-desc">Enter patient vitals & labs</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    age        = st.number_input("Age (years)", 20, 100, 55)
    sex        = st.selectbox("Sex", [0, 1], format_func=lambda x: ["Female","Male"][x])
    cp         = st.selectbox("Chest Pain Type", [0,1,2,3],
                              format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x])
    trestbps   = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol       = st.number_input("Cholesterol (mg/dL)", 100, 600, 220)
    fbs        = st.selectbox("Fasting BS > 120", [0,1], format_func=lambda x: ["No","Yes"][x])
    restecg    = st.selectbox("Resting ECG", [0,1,2],
                              format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
    thalach    = st.number_input("Max Heart Rate", 60, 220, 150)
    exang      = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: ["No","Yes"][x])
    oldpeak    = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    slope      = st.selectbox("ST Slope", [0,1,2],
                              format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
    ca         = st.selectbox("Major Vessels (CA)", [0,1,2,3,4])
    thal       = st.selectbox("Thalassemia", [0,1,2,3],
                              format_func=lambda x: ["Unknown","Normal","Fixed Defect","Reversible Defect"][x])
 
    if st.button("Run Clinical Analysis"):
        with st.spinner("Predicting..."):
            try:
                rf   = load_rf_model()
                data = np.array([[age,sex,cp,trestbps,chol,fbs,
                                  restecg,thalach,exang,oldpeak,slope,ca,thal]])
                prob = float(rf.predict_proba(data)[0][1])
                st.session_state.clin_risk = prob
                st.session_state.clin_prob = prob
 
                css = 'good' if prob < 0.4 else 'warn' if prob < 0.7 else 'bad'
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-chip {css}" style="flex:1">
                        <span class="val">{prob:.0%}</span>
                        <span class="lbl">Clinical Risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Clinical error: {e}")
 
# ─────────────────────────────────────────────
# FINAL RISK BLOCK
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
 
scores_available = [s for s in [
    st.session_state.ecg_risk,
    st.session_state.echo_risk,
    st.session_state.clin_risk
] if s is not None]
 
st.markdown("""
<div style="text-align:center;margin-bottom:1.2rem">
    <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f8f8f8;margin:0">
        Final Risk Assessment
    </h2>
    <p style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin:0.3rem 0 0">
        Weighted fusion of all available modules
    </p>
</div>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="weight-row" style="justify-content:center">
    <div class="weight-pill">ECG <b>30%</b></div>
    <div class="weight-pill">Echo <b>30%</b></div>
    <div class="weight-pill">Clinical <b>40%</b></div>
</div>
""", unsafe_allow_html=True)
 
if scores_available:
    # Weighted average — use only available modules, renormalise
    weights = {'ecg': 0.30, 'echo': 0.30, 'clin': 0.40}
    raw = {
        'ecg':  st.session_state.ecg_risk,
        'echo': st.session_state.echo_risk,
        'clin': st.session_state.clin_risk,
    }
    total_w = sum(weights[k] for k, v in raw.items() if v is not None)
    final   = sum(weights[k] * v for k, v in raw.items() if v is not None) / total_w
 
    cls   = 'green' if final < 0.4 else 'amber' if final < 0.7 else 'red'
    label = 'Low Risk' if final < 0.4 else 'Moderate Risk' if final < 0.7 else 'High Risk'
    advice = {
        'green': 'Maintain a heart-healthy lifestyle. Schedule routine annual check-ups.',
        'amber': 'Further diagnostic testing is advised. Consult a cardiologist.',
        'red':   'Immediate medical evaluation strongly recommended. Do not delay.'
    }[cls]
    icon  = {'green': '✓', 'amber': '⚠', 'red': '✕'}[cls]
 
    # Layout: gauge left, score right
    g_col, s_col = st.columns([1, 1])
 
    with g_col:
        fig_g = draw_gauge(final)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)
 
    with s_col:
        st.markdown(f"""
        <div class="result-block {cls}">
            <div class="result-score">{final:.0%}</div>
            <div class="result-label">{icon} {label}</div>
            <p class="result-advice">{advice}</p>
        </div>
        """, unsafe_allow_html=True)
 
    # Per-module breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
 
    def mini_chip(col, title, value, unit, css_class):
        col.markdown(f"""
        <div class="metric-chip {css_class}" style="text-align:center">
            <span class="val">{value}{unit}</span>
            <span class="lbl">{title}</span>
        </div>
        """, unsafe_allow_html=True)
 
    with m1:
        if st.session_state.ecg_risk is not None:
            r = st.session_state.ecg_risk
            mini_chip(m1, f"ECG · class {st.session_state.ecg_label}", f"{r:.0%}", "",
                      'good' if r < 0.4 else 'warn' if r < 0.7 else 'bad')
        else:
            st.markdown("<p style='color:rgba(255,255,255,0.25);font-size:0.8rem;text-align:center'>ECG not provided</p>", unsafe_allow_html=True)
 
    with m2:
        if st.session_state.echo_risk is not None:
            r = st.session_state.echo_risk
            mini_chip(m2, f"Echo · EF {st.session_state.ef_val:.0f}%", f"{r:.0%}", "",
                      'good' if r < 0.4 else 'warn' if r < 0.7 else 'bad')
        else:
            st.markdown("<p style='color:rgba(255,255,255,0.25);font-size:0.8rem;text-align:center'>Echo not provided</p>", unsafe_allow_html=True)
 
    with m3:
        if st.session_state.clin_risk is not None:
            r = st.session_state.clin_risk
            mini_chip(m3, "Clinical Model", f"{r:.0%}", "",
                      'good' if r < 0.4 else 'warn' if r < 0.7 else 'bad')
        else:
            st.markdown("<p style='color:rgba(255,255,255,0.25);font-size:0.8rem;text-align:center'>Clinical not run</p>", unsafe_allow_html=True)
 
else:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03);border:1px dashed rgba(255,255,255,0.1);
                border-radius:14px;padding:2.5rem;text-align:center">
        <p style="color:rgba(255,255,255,0.25);font-size:0.9rem;margin:0">
            Complete at least one module above to see the final risk score
        </p>
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<p style="font-size:0.72rem;color:rgba(255,255,255,0.2);text-align:center;margin-top:2.5rem">
    For research &amp; educational purposes only. Not a substitute for professional medical advice,
    diagnosis, or treatment. Always consult a qualified cardiologist.
</p>
""", unsafe_allow_html=True)
 