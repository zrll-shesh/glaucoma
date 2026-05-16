import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from PIL import Image

st.set_page_config(
    page_title="GlaucoStat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0A0A0A;
    color: #E8E4DC;
}

.main { background-color: #0A0A0A; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background-color: #0F0F0F;
    border-right: 1px solid #1E1E1E;
}
[data-testid="stSidebar"] * { color: #E8E4DC !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.82rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1E1E1E;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #555;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.75rem 1.5rem;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #E8E4DC !important;
    border-bottom: 2px solid #C8FF00 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem; }

.hero-wordmark {
    font-family: 'Instrument Serif', serif;
    font-size: 4.5rem;
    line-height: 1;
    letter-spacing: -0.02em;
    color: #E8E4DC;
}
.hero-wordmark span { color: #C8FF00; font-style: italic; }
.hero-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #C8FF00;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.metric-block {
    background: #111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #C8FF00;
}
.metric-number {
    font-family: 'Instrument Serif', serif;
    font-size: 2.8rem;
    color: #E8E4DC;
    line-height: 1;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #444;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.metric-sub {
    font-size: 0.72rem;
    color: #333;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}

.result-panel-pos {
    background: #0A1A0A;
    border: 1px solid #1A3A1A;
    border-left: 4px solid #4ADE80;
    border-radius: 4px;
    padding: 2rem;
}
.result-panel-neg {
    background: #1A0A0A;
    border: 1px solid #3A1A1A;
    border-left: 4px solid #F87171;
    border-radius: 4px;
    padding: 2rem;
}
.result-diagnosis {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    line-height: 1.1;
}
.result-prob {
    font-family: 'Instrument Serif', serif;
    font-size: 4rem;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-pos .result-diagnosis, .result-pos .result-prob { color: #F87171; }
.result-neg .result-diagnosis, .result-neg .result-prob { color: #4ADE80; }

.stat-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}
.stat-table tr { border-bottom: 1px solid #1A1A1A; }
.stat-table tr:last-child { border-bottom: none; }
.stat-table td { padding: 0.6rem 0; color: #888; }
.stat-table td:last-child { color: #E8E4DC; text-align: right; }

.quality-pill-high {
    display: inline-block;
    background: #0A1A0A;
    border: 1px solid #4ADE80;
    color: #4ADE80;
    padding: 0.2rem 0.75rem;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.quality-pill-medium {
    display: inline-block;
    background: #1A1400;
    border: 1px solid #FACC15;
    color: #FACC15;
    padding: 0.2rem 0.75rem;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.quality-pill-low {
    display: inline-block;
    background: #1A0A0A;
    border: 1px solid #F87171;
    color: #F87171;
    padding: 0.2rem 0.75rem;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}

.warning-box {
    background: #1A1400;
    border: 1px solid #FACC15;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    font-size: 0.8rem;
    color: #FACC15;
    font-family: 'JetBrains Mono', monospace;
}
.alert-box {
    background: #1A0A0A;
    border: 1px solid #F87171;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    font-size: 0.8rem;
    color: #F87171;
    font-family: 'JetBrains Mono', monospace;
}
.ok-box {
    background: #0A1A0A;
    border: 1px solid #4ADE80;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    font-size: 0.8rem;
    color: #4ADE80;
    font-family: 'JetBrains Mono', monospace;
}

.recommend-box {
    background: #111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.recommend-box h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #C8FF00;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.recommend-box p {
    font-size: 0.82rem;
    color: #666;
    line-height: 1.7;
}

.divider {
    border: none;
    border-top: 1px solid #1E1E1E;
    margin: 2rem 0;
}

.upload-zone {
    border: 1px dashed #2A2A2A;
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    background: #0F0F0F;
}
.upload-zone-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: #444;
    margin-bottom: 0.5rem;
}
.upload-zone-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #333;
    letter-spacing: 0.1em;
}

.sidebar-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.8rem;
    color: #E8E4DC;
    line-height: 1;
}
.sidebar-logo span { color: #C8FF00; font-style: italic; }
.sidebar-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #333;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}
.sidebar-model-card {
    background: #161616;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 1rem;
    margin-top: 1rem;
}
.sidebar-model-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #C8FF00;
    font-weight: 600;
}
.sidebar-model-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #444;
    margin-top: 0.2rem;
}

.stat-highlight {
    background: #111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.25rem;
}
.stat-highlight-val {
    font-family: 'Instrument Serif', serif;
    font-size: 1.8rem;
    color: #E8E4DC;
}
.stat-highlight-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #444;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.method-card {
    background: #111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.method-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #E8E4DC;
    margin-bottom: 0.35rem;
}
.method-card-desc {
    font-size: 0.8rem;
    color: #555;
    line-height: 1.7;
}

stFileUploader { background: transparent; }

.stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #444 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY HELPERS ────────────────────────────────────────────────
# Plotly 6.x breaking changes vs older versions:
#   1. 'transparent' is NOT a valid color — use 'rgba(0,0,0,0)'
#   2. Never put margin here — per-chart margin override must use a
#      separate update_layout() call to avoid duplicate kwarg error
#   3. Always style axes via update_xaxes() / update_yaxes(), never
#      inside the update_layout() dict
PLOTLY_LAYOUT = dict(
    plot_bgcolor='#111',
    paper_bgcolor='#111',
    font=dict(family='JetBrains Mono, monospace', size=11, color='#666'),
)

AXIS_STYLE = dict(
    showgrid=True,
    gridcolor='#1A1A1A',
    linecolor='#222',
    tickcolor='#333',
    zeroline=False,
)

LEGEND_STYLE = dict(
    bgcolor='rgba(0,0,0,0)',  # 'transparent' breaks Plotly 6.x
    font=dict(color='#555'),
)

DEFAULT_MARGIN = dict(l=50, r=20, t=40, b=50)


# ── ARTIFACT LOADING ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "glaucostat_artifacts"
    config_path = os.path.join(base, "model_config.json")
    model_path  = os.path.join(base, "best_model.keras")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None, None, None, None
    with open(config_path) as f:
        config = json.load(f)
    model  = keras.models.load_model(model_path)
    roc_df = pd.read_csv(os.path.join(base, "roc_data.csv"))
    cal_df = pd.read_csv(os.path.join(base, "calibration_data.csv"))
    return model, config, roc_df, cal_df


# ── IMAGE HELPERS ─────────────────────────────────────────────────
def preprocess_image(image_bytes, img_size=224):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = np.array(pil)[:, :, ::-1]
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_norm    = img_resized.astype(np.float32) / 255.0
    mean        = np.array([0.485, 0.456, 0.406])
    std         = np.array([0.229, 0.224, 0.225])
    img_pre     = (img_norm - mean) / std
    return img_pre, img_norm, img_rgb


def compute_gradcam(model, img_array, qs_array, conv_layer='top_conv'):
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            inputs = {
                'image_input':   tf.cast(img_array, tf.float32),
                'quality_input': tf.cast(qs_array,  tf.float32),
            }
            conv_out, preds = grad_model(inputs)
            loss = preds[:, 0]
        grads        = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_gradcam(img_norm, heatmap, alpha=0.45):
    h    = cv2.resize(heatmap, (img_norm.shape[1], img_norm.shape[0]))
    h    = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_INFERNO)
    h    = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    base = np.uint8(255 * np.clip(img_norm, 0, 1))
    return cv2.addWeighted(base, 1 - alpha, h, alpha, 0)


def get_quality_info(qs):
    if qs >= 5.0:
        return 'HIGH',   '#4ADE80', 'quality-pill-high',   'Kualitas gambar baik — analisis dapat dipercaya.'
    elif qs >= 3.0:
        return 'MEDIUM', '#FACC15', 'quality-pill-medium', 'Kualitas gambar sedang — hasil mungkin sedikit kurang akurat.'
    else:
        return 'LOW',    '#F87171', 'quality-pill-low',    'Kualitas gambar rendah — disarankan mengulang foto fundus.'


# ── LOAD ARTIFACTS ────────────────────────────────────────────────
model, config, roc_df, cal_df = load_artifacts()


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 0 1rem;">
        <div class="sidebar-logo">Glauco<span>Stat</span></div>
        <div class="sidebar-mono">Decision Support System v1.0</div>
    </div>
    <hr style="border-color: #1A1A1A; margin: 0 0 1.5rem;">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["Detection", "Statistical Validation", "Error Analysis", "About"],
        label_visibility="collapsed"
    )

    if config:
        st.markdown(f"""
        <div class="sidebar-model-card">
            <div class="sidebar-model-name">{config['backbone']}</div>
            <div class="sidebar-model-detail">Multimodal Transfer Learning</div>
            <div class="sidebar-model-detail" style="margin-top:0.5rem; color:#C8FF00;">
                AUC {config['bootstrap_auc']['mean']:.4f}
            </div>
            <div class="sidebar-model-detail">
                95% CI [{config['bootstrap_auc']['ci_lower']:.3f}, {config['bootstrap_auc']['ci_upper']:.3f}]
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        quality_score = st.slider(
            "QUALITY SCORE INPUT",
            min_value=1.0, max_value=8.0, value=5.5, step=0.1,
            help="FundusQ-Net quality score dari gambar fundus yang diupload."
        )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#2A2A2A; line-height:2;">
            DATASET — PhysioNet HYGD<br>
            THRESHOLD — {config['optimal_threshold']:.4f} (Youden)<br>
            TRAIN — {config['n_train']} images<br>
            TEST — {config['n_test']} images<br>
            GATE — QS &ge; 3.0
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: DETECTION ───────────────────────────────────────────────
if nav == "Detection":
    st.markdown("""
    <div style="padding: 1rem 0 2.5rem;">
        <div class="hero-wordmark">Glauco<span>Stat</span></div>
        <div class="hero-tagline">Quality-Aware Multimodal Transfer Learning — Gold-Standard PhysioNet HYGD</div>
    </div>
    """, unsafe_allow_html=True)

    if not model:
        st.error("Artefak model tidak ditemukan. Pastikan folder glaucostat_artifacts/ ada di direktori yang sama.")
        st.stop()

    st.markdown('<div class="section-eyebrow">Upload Fundus Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload fundus image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility="collapsed"
    )

    if uploaded:
        image_bytes = uploaded.read()
        img_pre, img_norm, img_rgb = preprocess_image(image_bytes, config['img_size'])

        qs      = quality_score
        qs_norm = (qs - config['qs_mean']) / config['qs_std']
        img_batch = np.expand_dims(img_pre, 0)
        qs_batch  = np.array([[qs_norm]])

        with st.spinner("Analyzing fundus image..."):
            pred = model.predict(
                {'image_input': img_batch, 'quality_input': qs_batch},
                verbose=0
            )[0][0]
            heatmap = compute_gradcam(model, img_batch, qs_batch, config.get('conv_layer', 'top_conv'))

        threshold   = config['optimal_threshold']
        is_glaucoma = pred >= threshold
        qs_tier, qs_color, qs_pill_class, qs_msg = get_quality_info(qs)

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        col_img, col_res = st.columns([1.1, 1], gap="large")

        with col_img:
            st.markdown('<div class="section-eyebrow">Fundus Image — Grad-CAM</div>', unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["ORIGINAL", "HEATMAP", "OVERLAY"])
            with tab1:
                st.image(img_rgb, use_container_width=True)
            with tab2:
                if heatmap is not None:
                    fig_hm, ax_hm = plt.subplots(figsize=(5, 5), facecolor='#111')
                    ax_hm.imshow(heatmap, cmap='inferno')
                    ax_hm.axis('off')
                    ax_hm.set_facecolor('#111')
                    plt.tight_layout(pad=0)
                    st.pyplot(fig_hm, use_container_width=True)
                    plt.close()
                else:
                    st.info("Grad-CAM tidak tersedia untuk gambar ini.")
            with tab3:
                if heatmap is not None:
                    overlay = overlay_gradcam(img_norm, heatmap)
                    st.image(overlay, use_container_width=True)
                else:
                    st.image(img_rgb, use_container_width=True)

        with col_res:
            st.markdown('<div class="section-eyebrow">Diagnosis Result</div>', unsafe_allow_html=True)

            if is_glaucoma:
                st.markdown(f"""
                <div class="result-panel-neg result-pos">
                    <div class="result-diagnosis">Glaucomatous<br>Optic Neuropathy</div>
                    <div class="result-prob">{pred*100:.1f}%</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#F87171; letter-spacing:0.1em;">
                        PROBABILITY — GON POSITIVE
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-panel-pos result-neg">
                    <div class="result-diagnosis">No Glaucoma<br>Detected</div>
                    <div class="result-prob">{(1-pred)*100:.1f}%</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#4ADE80; letter-spacing:0.1em;">
                        PROBABILITY — GON NEGATIVE
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

            if qs < 3.0:
                st.markdown(f'<div class="alert-box">QS {qs:.1f} — {qs_msg}</div>', unsafe_allow_html=True)
            elif qs < 5.0:
                st.markdown(f'<div class="warning-box">QS {qs:.1f} — {qs_msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-box">QS {qs:.1f} — {qs_msg}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-highlight">
                <table class="stat-table">
                    <tr><td>Raw Probability</td><td>{pred:.4f}</td></tr>
                    <tr><td>Threshold (Youden)</td><td>{threshold:.4f}</td></tr>
                    <tr><td>Quality Score</td><td><span class="{qs_pill_class}">{qs:.1f} {qs_tier}</span></td></tr>
                    <tr><td>Model AUC</td><td>{config['bootstrap_auc']['mean']:.4f}</td></tr>
                    <tr><td>95% CI</td><td>[{config['bootstrap_auc']['ci_lower']:.3f}, {config['bootstrap_auc']['ci_upper']:.3f}]</td></tr>
                    <tr><td>Sensitivity</td><td>{config['sensitivity']:.4f}</td></tr>
                    <tr><td>Specificity</td><td>{config['specificity']:.4f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

            if is_glaucoma:
                st.markdown("""
                <div class="recommend-box">
                    <h4>Clinical Recommendation</h4>
                    <p>Sistem mendeteksi indikasi Glaucomatous Optic Neuropathy (GON+). Pasien disarankan segera dirujuk ke dokter spesialis mata untuk konfirmasi melalui tonometri, perimetri 24-2, dan OCT RNFL. Jangan tunda rujukan.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommend-box">
                    <h4>Clinical Recommendation</h4>
                    <p>Tidak terdeteksi indikasi glaukoma pada citra ini. Tetap lakukan pemeriksaan mata rutin setiap 1-2 tahun. Pasien dengan faktor risiko (riwayat keluarga, IOP tinggi, usia >60 tahun) disarankan pemeriksaan lebih sering.</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-zone-title">Drop fundus image here</div>
            <div class="upload-zone-sub">JPG / PNG / BMP — Retinal fundus photograph</div>
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: STATISTICAL VALIDATION ─────────────────────────────────
elif nav == "Statistical Validation":
    st.markdown("""
    <div style="padding: 1rem 0 2rem;">
        <div class="section-eyebrow">Model Performance</div>
        <div style="font-family:'Instrument Serif',serif; font-size:2.8rem; line-height:1.1; color:#E8E4DC;">
            Statistical<br>Validation
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; margin-top:0.5rem; letter-spacing:0.1em;">
            Bootstrapped AUC · Youden Index · Calibration · Spearman · McNemar · AUC per Tier
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not config:
        st.error("Artefak tidak ditemukan.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("AUC-ROC",      f"{config['bootstrap_auc']['mean']:.4f}", f"CI [{config['bootstrap_auc']['ci_lower']:.3f}, {config['bootstrap_auc']['ci_upper']:.3f}]"),
        ("Sensitivity",  f"{config['sensitivity']:.4f}",           "True Positive Rate"),
        ("Specificity",  f"{config['specificity']:.4f}",           "True Negative Rate"),
        ("Brier Score",  f"{config['brier_score']:.4f}",           "Calibration quality"),
        ("Youden Index", f"{config['youden_index']:.4f}",          f"Threshold {config['optimal_threshold']:.4f}"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-number">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    # ── ROC Curve ─────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-eyebrow">ROC Curve — Bootstrapped 95% CI</div>', unsafe_allow_html=True)
        if roc_df is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_df['fpr'], y=roc_df['tpr'],
                mode='lines', name='GlaucoStat',
                line=dict(color='#C8FF00', width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(color='#2A2A2A', dash='dash', width=1),
                name='Random', showlegend=True
            ))
            fig_roc.add_trace(go.Scatter(
                x=[1 - config['specificity']], y=[config['sensitivity']],
                mode='markers', name='Youden Point',
                marker=dict(color='#F87171', size=10, symbol='circle')
            ))
            # Two-step: base props first, then margin separately
            fig_roc.update_layout(**PLOTLY_LAYOUT, height=340, legend=LEGEND_STYLE)
            fig_roc.update_layout(margin=DEFAULT_MARGIN)
            fig_roc.update_xaxes(**AXIS_STYLE, title_text='False Positive Rate')
            fig_roc.update_yaxes(**AXIS_STYLE, title_text='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)

    # ── Calibration Curve ─────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-eyebrow">Calibration Curve — Brier Score Analysis</div>', unsafe_allow_html=True)
        if cal_df is not None:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(color='#2A2A2A', dash='dash', width=1),
                name='Perfect Calibration'
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal_df['mean_pred'], y=cal_df['fraction_pos'],
                mode='lines+markers',
                name=f"GlaucoStat (Brier={config['brier_score']:.4f})",
                line=dict(color='#C8FF00', width=2),
                marker=dict(color='#E8E4DC', size=7)
            ))
            fig_cal.update_layout(**PLOTLY_LAYOUT, height=340, legend=LEGEND_STYLE)
            fig_cal.update_layout(margin=DEFAULT_MARGIN)
            fig_cal.update_xaxes(**AXIS_STYLE, title_text='Mean Predicted Probability')
            fig_cal.update_yaxes(**AXIS_STYLE, title_text='Fraction of Positives')
            st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3, gap="large")

    # ── AUC per Quality Tier ──────────────────────────────────────
    with col_a:
        st.markdown('<div class="section-eyebrow">AUC per Quality Tier</div>', unsafe_allow_html=True)
        if config.get('tier_auc'):
            tiers  = list(config['tier_auc'].keys())
            values = list(config['tier_auc'].values())
            colors = ['#FACC15', '#4ADE80']
            fig_tier = go.Figure(go.Bar(
                x=tiers, y=values,
                marker_color=colors[:len(tiers)],
                text=[f"{v:.4f}" for v in values],
                textposition='outside',
                textfont=dict(color='#E8E4DC', family='JetBrains Mono'),
                width=0.4
            ))
            fig_tier.update_layout(**PLOTLY_LAYOUT, height=280)
            fig_tier.update_layout(margin=DEFAULT_MARGIN)
            fig_tier.update_xaxes(**AXIS_STYLE)
            fig_tier.update_yaxes(**AXIS_STYLE, range=[0.4, 1.0])
            st.plotly_chart(fig_tier, use_container_width=True)

    # ── Spearman ──────────────────────────────────────────────────
    with col_b:
        st.markdown('<div class="section-eyebrow">Spearman Correlation</div>', unsafe_allow_html=True)
        rho   = config['spearman_rho']
        p_val = config['spearman_p']
        sig   = p_val < 0.05
        st.markdown(f"""
        <div class="stat-highlight" style="margin-top:0;">
            <div class="stat-highlight-val">{rho:.4f}</div>
            <div class="stat-highlight-label">Spearman rho — QS vs Confidence</div>
            <div style="height:1rem;"></div>
            <table class="stat-table">
                <tr><td>p-value</td><td>{p_val:.4f}</td></tr>
                <tr><td>Alpha</td><td>0.05</td></tr>
                <tr><td>H0</td><td>{'REJECTED' if sig else 'NOT REJECTED'}</td></tr>
                <tr><td>Result</td><td style="color:{'#4ADE80' if sig else '#F87171'};">{'Significant' if sig else 'Not Significant'}</td></tr>
            </table>
            <div style="height:0.75rem;"></div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; line-height:1.6;">
                {'QS berkorelasi signifikan dengan confidence prediksi model.' if sig else 'Tidak ditemukan korelasi signifikan antara QS dan confidence.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── McNemar ───────────────────────────────────────────────────
    with col_c:
        st.markdown('<div class="section-eyebrow">McNemar Test</div>', unsafe_allow_html=True)
        mc_p          = config['mcnemar_pvalue']
        mc_sig        = mc_p < 0.05
        tier_auc_vals = config.get('tier_auc', {})
        high_auc      = tier_auc_vals.get('High (>5)',    0)
        medium_auc    = tier_auc_vals.get('Medium (3-5)', 0)
        st.markdown(f"""
        <div class="stat-highlight" style="margin-top:0;">
            <div class="stat-highlight-val" style="color:{'#4ADE80' if mc_sig else '#F87171'};">
                {'Significant' if mc_sig else 'Not Sig.'}
            </div>
            <div class="stat-highlight-label">High vs Medium Quality — Error Pattern</div>
            <div style="height:1rem;"></div>
            <table class="stat-table">
                <tr><td>p-value</td><td>{mc_p:.4f}</td></tr>
                <tr><td>AUC High (&gt;5)</td><td>{high_auc:.4f}</td></tr>
                <tr><td>AUC Medium (3-5)</td><td>{medium_auc:.4f}</td></tr>
                <tr><td>AUC Gap</td><td>{high_auc - medium_auc:.4f}</td></tr>
            </table>
            <div style="height:0.75rem;"></div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; line-height:1.6;">
                {'Error pattern berbeda signifikan antar quality tier — kualitas gambar terbukti mempengaruhi akurasi model.' if mc_sig else 'Tidak ada perbedaan signifikan.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Complete Statistical Summary</div>', unsafe_allow_html=True)

    summary_path = "glaucostat_artifacts/statistical_summary.csv"
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metrik"      : st.column_config.TextColumn("Metric",         width="medium"),
                "Nilai"       : st.column_config.TextColumn("Value",          width="small"),
                "Interpretasi": st.column_config.TextColumn("Interpretation", width="large"),
            }
        )


# ── PAGE: ERROR ANALYSIS ──────────────────────────────────────────
elif nav == "Error Analysis":
    st.markdown("""
    <div style="padding: 1rem 0 2rem;">
        <div class="section-eyebrow">Model Failure Mode</div>
        <div style="font-family:'Instrument Serif',serif; font-size:2.8rem; line-height:1.1; color:#E8E4DC;">
            Error<br>Analysis
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; margin-top:0.5rem; letter-spacing:0.1em;">
            False Positive · False Negative · MC Dropout Flag Rate · Clinical Risk
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not config:
        st.error("Artefak tidak ditemukan.")
        st.stop()

    cm    = config['confusion_matrix']
    tp    = cm['tp']
    tn    = cm['tn']
    fp    = cm['fp']
    fn    = cm['fn']
    total = tp + tn + fp + fn

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in zip(
        [c1, c2, c3, c4],
        ['True Positive', 'True Negative', 'False Positive', 'False Negative (Kritis)'],
        [tp, tn, fp, fn],
        ['#4ADE80', '#4ADE80', '#FACC15', '#F87171']
    ):
        with col:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-number" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{val/total*100:.1f}% of test set</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    # ── MC Dropout Flag Rate ──────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-eyebrow">MC Dropout Flag Rate per Error Type</div>', unsafe_allow_html=True)
        fn_flag  = config['fn_flag_rate'] * 100
        fp_flag  = 50.0
        cor_flag = 47.5

        fig_flag = go.Figure(go.Bar(
            x=['False Negative\n(Kritis)', 'False Positive', 'Correct'],
            y=[fn_flag, fp_flag, cor_flag],
            marker_color=['#F87171', '#FACC15', '#4ADE80'],
            text=[f"{fn_flag:.1f}%", f"{fp_flag:.1f}%", f"{cor_flag:.1f}%"],
            textposition='outside',
            textfont=dict(color='#E8E4DC', family='JetBrains Mono'),
            width=0.45
        ))
        fig_flag.update_layout(**PLOTLY_LAYOUT, height=320)
        fig_flag.update_layout(margin=DEFAULT_MARGIN)
        fig_flag.update_xaxes(**AXIS_STYLE)
        fig_flag.update_yaxes(**AXIS_STYLE, range=[0, 100], title_text='% Diflag (uncertainty >= 0.20)')
        st.plotly_chart(fig_flag, use_container_width=True)

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#555; line-height:1.8; margin-top:0.5rem;">
            {fn_flag:.1f}% kasus False Negative (glaukoma yang terlewat) memiliki uncertainty >= 0.20
            dan dapat diflag secara otomatis untuk review manual dokter — mengurangi risiko klinis
            tanpa menambah beban kerja berlebihan pada kasus yang sudah yakin.
        </div>
        """, unsafe_allow_html=True)

    # ── Confusion Matrix ──────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-eyebrow">Confusion Matrix Heatmap</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=['Predicted GON-', 'Predicted GON+'],
            y=['True GON-',      'True GON+'],
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate='%{text}',
            textfont=dict(size=20, color='#E8E4DC', family='Instrument Serif'),
            colorscale=[[0, '#111'], [1, '#1A3A1A']],
            showscale=False
        ))
        # Confusion matrix uses a custom margin — apply in a separate call
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=320)
        fig_cm.update_layout(margin=dict(l=80, r=20, t=20, b=80))
        fig_cm.update_xaxes(**AXIS_STYLE)
        fig_cm.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Clinical Risk Interpretation</div>', unsafe_allow_html=True)

    col_fn, col_fp = st.columns(2, gap="large")
    with col_fn:
        st.markdown(f"""
        <div class="stat-highlight" style="border-left: 3px solid #F87171;">
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#F87171; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
                False Negative — n={fn}
            </div>
            <table class="stat-table">
                <tr><td>Proporsi dari test set</td><td>{fn/total*100:.1f}%</td></tr>
                <tr><td>Risiko klinis</td><td style="color:#F87171;">TINGGI</td></tr>
                <tr><td>Implikasi</td><td>Pasien GON+ terlewat</td></tr>
                <tr><td>Diflag MC Dropout</td><td>{config['fn_flag_rate']*100:.1f}% kasus</td></tr>
            </table>
            <div style="height:0.75rem;"></div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; line-height:1.6;">
                Pasien glaukoma yang tidak terdeteksi sistem. Dalam konteks skrining, ini adalah error paling kritis
                karena menyebabkan keterlambatan penanganan dan risiko kebutaan permanen.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_fp:
        st.markdown(f"""
        <div class="stat-highlight" style="border-left: 3px solid #FACC15;">
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#FACC15; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
                False Positive — n={fp}
            </div>
            <table class="stat-table">
                <tr><td>Proporsi dari test set</td><td>{fp/total*100:.1f}%</td></tr>
                <tr><td>Risiko klinis</td><td style="color:#FACC15;">SEDANG</td></tr>
                <tr><td>Implikasi</td><td>Rujukan tidak perlu</td></tr>
                <tr><td>Dapat diminimalkan</td><td>Dengan quality gate</td></tr>
            </table>
            <div style="height:0.75rem;"></div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; line-height:1.6;">
                Pasien normal yang diprediksi glaukoma. Menyebabkan rujukan tidak perlu ke spesialis,
                namun tidak mengancam jiwa secara langsung. Lebih dapat ditoleransi daripada False Negative.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    grad_path = "glaucostat_artifacts/gradcam_results.png"
    if os.path.exists(grad_path):
        st.markdown('<div class="section-eyebrow">Grad-CAM Visualization — Sample Cases</div>', unsafe_allow_html=True)
        st.image(grad_path, use_container_width=True)


# ── PAGE: ABOUT ───────────────────────────────────────────────────
elif nav == "About":
    st.markdown("""
    <div style="padding: 1rem 0 2rem;">
        <div class="section-eyebrow">Research Background</div>
        <div style="font-family:'Instrument Serif',serif; font-size:2.8rem; line-height:1.1; color:#E8E4DC;">
            About<br>GlaucoStat
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#444; margin-top:0.5rem; letter-spacing:0.1em;">
            Statistical Essay Competition — Satria Data 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-eyebrow">Background</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem; color:#555; line-height:1.9;">
            Glaukoma (Glaucomatous Optic Neuropathy / GON) adalah penyebab kebutaan ireversibel terbesar
            kedua di dunia dengan estimasi 64.3 juta penderita secara global dan diproyeksikan mencapai
            111.8 juta pada 2040. Sekitar 50% kasus tidak terdiagnosis hingga stadium lanjut.
            <br><br>
            Di Indonesia, ketimpangan distribusi dokter spesialis mata menyebabkan skrining dini tidak
            terjangkau bagi masyarakat di daerah terpencil. GlaucoStat hadir sebagai decision support system
            berbasis AI untuk tenaga kesehatan di fasilitas kesehatan primer.
            <br><br>
            Keunggulan utama GlaucoStat dibanding penelitian serupa adalah penggunaan
            <strong style="color:#E8E4DC;">gold-standard annotation</strong> dari HYGD PhysioNet —
            satu-satunya dataset glaukoma publik dengan label berdasarkan pemeriksaan klinis lengkap
            (Visual Field Test + IOP + OCT), bukan sekadar interpretasi foto fundus.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-eyebrow">Methodology</div>', unsafe_allow_html=True)
        methods = [
            ("Quality Gate & Stratification",
             "Filter gambar QS < 3.0 menggunakan FundusQ-Net score. Stratifikasi ke tier Medium (3-5) dan High (>5) untuk analisis statistik berbasis kualitas."),
            ("Patient-Level Split",
             "Stratified split 80:20 berbasis Patient ID — satu pasien tidak muncul di train dan test sekaligus. Mencegah data leakage yang umum diabaikan penelitian lain."),
            ("Quality-Aware Multimodal Transfer Learning",
             "EfficientNetB0 pretrained ImageNet + Quality Score sebagai scalar input. Keduanya di-fuse di fully connected layer — model belajar menyesuaikan confidence berdasarkan kualitas gambar."),
            ("Monte Carlo Dropout",
             "30 iterasi inference dengan dropout aktif untuk epistemic uncertainty estimation. Flag otomatis kasus berisiko tinggi untuk review manual dokter."),
            ("Comprehensive Statistical Validation",
             "Bootstrapped AUC 95% CI (1000 iter), Youden Index, McNemar Test, Spearman Correlation, Calibration Curve, Brier Score, Error Analysis."),
        ]
        for title, desc in methods:
            st.markdown(f"""
            <div class="method-card">
                <div class="method-card-title">{title}</div>
                <div class="method-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Dataset & Citation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-highlight">
        <table class="stat-table">
            <tr><td>Dataset</td><td>Hillel Yaffe Glaucoma Dataset (HYGD)</td></tr>
            <tr><td>Source</td><td>PhysioNet v1.0.0</td></tr>
            <tr><td>DOI</td><td>10.13026/z0ak-km33</td></tr>
            <tr><td>Total Images</td><td>747 (548 GON+, 199 GON-)</td></tr>
            <tr><td>Annotation</td><td>Gold-standard — VF + IOP + OCT + 1yr follow-up</td></tr>
            <tr><td>Camera</td><td>TOPCON DRI OCT Triton, 45° FOV</td></tr>
            <tr><td>QS Method</td><td>FundusQ-Net (scale 1-10, mean 5.9 ± 1.0)</td></tr>
            <tr><td>Ethics</td><td>Helsinki Committee HYMC-0029-24</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#2A2A2A; line-height:2;">
        Disclaimer: Sistem ini adalah prototipe untuk keperluan riset dan kompetisi ilmiah.
        Tidak menggantikan diagnosis dokter spesialis mata. Seluruh hasil prediksi harus dikonfirmasi
        oleh tenaga medis yang berkompeten sebelum pengambilan keputusan klinis.
    </div>
    """, unsafe_allow_html=True)
