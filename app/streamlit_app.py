"""
Streamlit Verification App — Multimodal Fake News Detector
=============================================================
Interactive web application that:
    1. Accepts an uploaded image
    2. Extracts text from the image using OCR (EasyOCR)
    3. Runs the multimodal fake news detection model
    4. Displays prediction results with rich visualizations

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import time
import base64
import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom CSS for Premium Aesthetics
# =============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* ---- Global Theming ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
            font-family: 'Inter', sans-serif;
        }

        /* ---- Main Title ---- */
        .main-title {
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .sub-title {
            text-align: center;
            color: #8892b0;
            font-size: 1.05rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* ---- Glassmorphism Cards ---- */
        .glass-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.75rem 0;
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(102, 126, 234, 0.3);
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        }

        /* ---- Result Cards ---- */
        .result-real {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.12) 0%, rgba(76, 175, 80, 0.04) 100%);
            border: 1px solid rgba(76, 175, 80, 0.3);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }
        .result-fake {
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.12) 0%, rgba(244, 67, 54, 0.04) 100%);
            border: 1px solid rgba(244, 67, 54, 0.3);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        .result-label {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .result-label-real { color: #4CAF50; }
        .result-label-fake { color: #F44336; }

        .confidence-text {
            font-size: 1.1rem;
            color: #b0b8d1;
            font-weight: 400;
        }

        /* ---- Upload Area ---- */
        .upload-area {
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            background: rgba(102, 126, 234, 0.04);
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: rgba(102, 126, 234, 0.7);
            background: rgba(102, 126, 234, 0.08);
        }

        /* ---- Extracted Text Area ---- */
        .extracted-text-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.2rem;
            color: #ccd6f6;
            font-size: 0.95rem;
            line-height: 1.7;
            max-height: 300px;
            overflow-y: auto;
        }

        /* ---- Section Headers ---- */
        .section-header {
            color: #ccd6f6;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 1.5rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* ---- Metric Tags ---- */
        .metric-tag {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0.25rem;
        }
        .metric-tag-blue {
            background: rgba(33, 150, 243, 0.15);
            color: #64B5F6;
            border: 1px solid rgba(33, 150, 243, 0.3);
        }
        .metric-tag-purple {
            background: rgba(156, 39, 176, 0.15);
            color: #CE93D8;
            border: 1px solid rgba(156, 39, 176, 0.3);
        }
        .metric-tag-green {
            background: rgba(76, 175, 80, 0.15);
            color: #81C784;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        /* ---- Pipeline Step Badges ---- */
        .pipeline-step {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 500;
            color: #ccd6f6;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .pipeline-arrow {
            color: #667eea;
            font-size: 1.2rem;
            margin: 0 0.3rem;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        /* ---- Progress animation ---- */
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .shimmer-text {
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s linear infinite;
        }

        /* ---- Hide Streamlit branding ---- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ---- Button styling ---- */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        /* ---- Divider ---- */
        .custom-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
            margin: 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# OCR: Text Extraction from Image (EasyOCR)
# =============================================================================
@st.cache_resource
def load_ocr_reader():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    return reader

def extract_text_from_image(image: Image.Image) -> dict:
    """
    Extract text from an image using EasyOCR.

    Returns:
        dict with 'text', 'segments', 'avg_confidence', 'raw_results'
    """
    reader = load_ocr_reader()

    # Convert PIL to numpy
    img_array = np.array(image)

    # Run OCR
    # EasyOCR format: [ [bbox, text, conf], ... ]
    lines = reader.readtext(img_array)

    if not lines:
        return {
            "text": "",
            "segments": [],
            "avg_confidence": 0.0,
            "raw_results": [],
        }

    segments = []
    confidences = []
    raw_results = []

    for line in lines:
        bbox = line[0]          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1]
        conf = float(line[2])

        segments.append({
            "text": text,
            "confidence": conf,
            "bbox": bbox,
        })
        confidences.append(conf)

        # keep same shape your draw_ocr_boxes expects: (bbox, text, conf)
        raw_results.append((bbox, text, conf))

    full_text = " ".join(s["text"] for s in segments)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": full_text,
        "segments": segments,
        "avg_confidence": avg_confidence,
        "raw_results": raw_results,
    }


def draw_ocr_boxes(image: Image.Image, raw_results: list) -> Image.Image:
    """
    Draw OCR detection boxes and labels on a copy of the input image.

    Args:
        image: Input PIL image.
        raw_results: List of tuples shaped as (bbox, text, confidence).

    Returns:
        Annotated PIL image.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for item in raw_results:
        if not item or len(item) != 3:
            continue

        bbox, text, conf = item
        if not bbox or len(bbox) < 4:
            continue

        points = [(int(p[0]), int(p[1])) for p in bbox]
        color = "#4CAF50" if conf >= 0.8 else "#FF9800" if conf >= 0.5 else "#F44336"

        draw.line(points + [points[0]], fill=color, width=3)

        label = f"{text[:30]} ({conf:.2f})"
        x, y = points[0]
        draw.rectangle([(x, max(0, y - 16)), (x + len(label) * 7, y)], fill=color)
        draw.text((x + 2, max(0, y - 14)), label, fill="white", font=font)

    return annotated



# =============================================================================
# Model Loading and Prediction
# =============================================================================
@st.cache_resource
def load_model(checkpoint_path: str):
    """Load the multimodal fake news detector."""
    try:
        from src.inference.predictor import MultimodalPredictor
        import torch

        if os.path.exists(checkpoint_path):
            predictor = MultimodalPredictor.from_checkpoint(
                checkpoint_path,
                device=torch.device("cpu"),
            )
            return predictor, True
        else:
            return None, False
    except Exception as e:
        st.warning(f"Model loading error: {e}")
        return None, False


def run_prediction(predictor, text: str, image: Image.Image) -> dict:
    """Run model prediction on text + image."""
    try:
        result = predictor.predict(text=text, image=image, mode="multimodal")
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def simulate_prediction(text: str) -> dict:
    """
    Simulate a prediction when no model is trained yet.
    Uses heuristic rules for demonstration purposes.
    """
    import hashlib
    import re

    text_lower = text.lower().strip()
    fake_score = 0.5  # Start neutral

    # Heuristic indicators of fake news
    fake_indicators = [
        "breaking", "shocking", "you won't believe", "secret", "exposed",
        "they don't want you to know", "miracle", "hoax", "conspiracy",
        "100%", "guaranteed", "urgent", "act now", "share before deleted",
        "forward this", "must see", "going viral", "mainstream media won't",
        "wake up", "what they're hiding", "exposed", "scandal",
    ]
    real_indicators = [
        "according to", "researchers", "study", "published", "university",
        "scientists", "report", "official", "data shows", "evidence",
        "peer-reviewed", "analysis", "findings", "source", "confirmed",
    ]

    for indicator in fake_indicators:
        if indicator in text_lower:
            fake_score += 0.08

    for indicator in real_indicators:
        if indicator in text_lower:
            fake_score -= 0.06

    # Exclamation marks bias
    exclamation_count = text.count("!")
    fake_score += min(exclamation_count * 0.04, 0.15)

    # ALL CAPS words bias
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    fake_score += caps_ratio * 0.2

    # Very short text is suspicious
    if len(text.split()) < 5 and len(text) > 0:
        fake_score += 0.05

    # Clamp to [0.05, 0.95]
    fake_score = max(0.05, min(0.95, fake_score))
    real_score = 1.0 - fake_score

    pred_class = 1 if fake_score > 0.5 else 0
    confidence = fake_score if pred_class == 1 else real_score

    return {
        "prediction": "Fake" if pred_class == 1 else "Real",
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": {
            "real": real_score,
            "fake": fake_score,
        },
    }


# =============================================================================
# Plotly Visualization Components
# =============================================================================
def create_confidence_gauge(real_prob: float, fake_prob: float, prediction: str) -> go.Figure:
    """Create a dual-gauge confidence meter."""
    fig = go.Figure()

    # Fake probability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=fake_prob * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#F44336" if fake_prob > 0.5 else "#888"}},
        title={"text": "Fake Probability", "font": {"size": 14, "color": "#ccc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "dtick": 25},
            "bar": {"color": "#F44336" if fake_prob > 0.5 else "#FF9800"},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(76,175,80,0.15)"},
                {"range": [30, 60], "color": "rgba(255,193,7,0.12)"},
                {"range": [60, 100], "color": "rgba(244,67,54,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 2},
                "thickness": 0.8,
                "value": fake_prob * 100,
            },
        },
        domain={"x": [0, 0.48], "y": [0, 1]},
    ))

    # Real probability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=real_prob * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#4CAF50" if real_prob > 0.5 else "#888"}},
        title={"text": "Real Probability", "font": {"size": 14, "color": "#ccc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "dtick": 25},
            "bar": {"color": "#4CAF50" if real_prob > 0.5 else "#FF9800"},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(244,67,54,0.15)"},
                {"range": [30, 60], "color": "rgba(255,193,7,0.12)"},
                {"range": [60, 100], "color": "rgba(76,175,80,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 2},
                "thickness": 0.8,
                "value": real_prob * 100,
            },
        },
        domain={"x": [0.52, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )

    return fig


def create_probability_bar(real_prob: float, fake_prob: float) -> go.Figure:
    """Create a horizontal stacked probability bar."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Prediction"],
        x=[real_prob * 100],
        orientation="h",
        name="Real",
        marker=dict(color="#4CAF50", line=dict(width=0)),
        text=[f"Real: {real_prob:.1%}"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="Inter"),
    ))

    fig.add_trace(go.Bar(
        y=["Prediction"],
        x=[fake_prob * 100],
        orientation="h",
        name="Fake",
        marker=dict(color="#F44336", line=dict(width=0)),
        text=[f"Fake: {fake_prob:.1%}"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="Inter"),
    ))

    fig.update_layout(
        barmode="stack",
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100], showticklabels=False, showgrid=False, zeroline=False
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
    )

    return fig


def create_ocr_confidence_chart(segments: list) -> go.Figure:
    """Create a bar chart of OCR segment confidences."""
    if not segments:
        return None

    texts = [s["text"][:25] + ("..." if len(s["text"]) > 25 else "") for s in segments]
    confs = [s["confidence"] * 100 for s in segments]

    colors = [
        "#4CAF50" if c > 80 else "#FF9800" if c > 50 else "#F44336"
        for c in confs
    ]

    fig = go.Figure(go.Bar(
        x=confs,
        y=texts,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:.0f}%" for c in confs],
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))

    fig.update_layout(
        title_text="OCR Segment Confidence",
        title_font=dict(size=14, color="#ccc"),
        height=max(200, len(segments) * 35 + 60),
        margin=dict(l=10, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    fig.update_xaxes(
        range=[0, 105],
        title_text="Confidence %",
        title_font=dict(color="#888"),
        tickfont=dict(color="#888"),
        gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(color="#ccc", size=10),
    )

    return fig


# =============================================================================
# Main Application
# =============================================================================
def main():
    inject_custom_css()

    # ---- Header ----
    st.markdown('<h1 class="main-title">🔍 Multimodal Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Upload an image → Extract text via OCR → Analyze with AI → Get verification results</p>',
        unsafe_allow_html=True,
    )

    # ---- Pipeline indicator ----
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 1.5rem;">
        <span class="pipeline-step">📷 Image Upload</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">📝 OCR Extraction</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">🧠 BERT + ViT</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">🔗 Cross-Modal Fusion</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">✅ Verdict</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Model checkpoint
        checkpoint_path = st.text_input(
            "Model Checkpoint",
            value="combined_model_artifacts/checkpoints/best_model_combined.pt",
            help="Path to trained model checkpoint (.pt file)",
        )

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Detection threshold
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for classifying as Fake",
        )

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # OCR language
        st.markdown("### 🌐 OCR Settings")
        ocr_lang = st.selectbox("OCR Language", ["English"], index=0)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Model info
        st.markdown("### 📊 Model Info")
        model_exists = os.path.exists(checkpoint_path)
        if model_exists:
            st.success("✅ Model checkpoint found")
        else:
            st.info("ℹ️ No checkpoint found — running in **demo mode** with heuristic analysis")

        st.markdown("""
        <div class="glass-card" style="font-size: 0.82rem; color: #8892b0;">
            <strong style="color: #ccd6f6;">Architecture</strong><br>
            • Text: BERT-base-uncased<br>
            • Image: ViT-base-patch16-224<br>
            • Fusion: Cross-Modal Attention (×2)<br>
            • Classifier: 512 → 256 → 128 → 2
        </div>
        """, unsafe_allow_html=True)

    # ---- Main Area ----
    col_upload, col_spacer, col_result = st.columns([5, 0.5, 5])

    with col_upload:
        st.markdown('<div class="section-header">📷 Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a news image, screenshot, or social media post",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, PNG, WebP, BMP",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Image metadata
            width, height = image.size
            file_size = uploaded_file.size / 1024  # KB
            st.markdown(f"""
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                <span class="metric-tag metric-tag-blue">📐 {width}×{height}px</span>
                <span class="metric-tag metric-tag-purple">💾 {file_size:.1f} KB</span>
                <span class="metric-tag metric-tag-green">🖼️ {uploaded_file.type}</span>
            </div>
            """, unsafe_allow_html=True)

    # ---- Analysis ----
    if uploaded_file is not None:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Analyze button
        analyze_col1, analyze_col2, analyze_col3 = st.columns([2, 1, 2])
        with analyze_col2:
            analyze_btn = st.button("🚀 Analyze", use_container_width=True)

        if analyze_btn:
            # Step 1: OCR Text Extraction
            with st.status("🔍 Analyzing image...", expanded=True) as status:
                st.write("📝 **Step 1/3**: Extracting text from image via OCR...")
                progress = st.progress(0)

                ocr_start = time.time()
                ocr_result = extract_text_from_image(image)
                ocr_time = time.time() - ocr_start
                progress.progress(33)

                extracted_text = ocr_result["text"]

                st.write(f"✅ OCR complete — extracted **{len(ocr_result['segments'])}** text segments in **{ocr_time:.1f}s**")

                # Step 2: Run Prediction
                st.write("🧠 **Step 2/3**: Running multimodal analysis (BERT + ViT + Cross-Modal Attention)...")
                progress.progress(66)

                pred_start = time.time()

                # Try loading trained model, fall back to simulation
                if os.path.exists(checkpoint_path):
                    predictor, loaded = load_model(checkpoint_path)
                    if loaded and predictor:
                        prediction = run_prediction(predictor, extracted_text, image)
                        mode_label = "Model"
                    else:
                        prediction = simulate_prediction(extracted_text)
                        mode_label = "Demo (Heuristic)"
                else:
                    prediction = simulate_prediction(extracted_text)
                    mode_label = "Demo (Heuristic)"

                pred_time = time.time() - pred_start
                progress.progress(90)

                st.write(f"✅ Prediction complete in **{pred_time:.2f}s** — Mode: **{mode_label}**")

                # Step 3: Generate visualizations
                st.write("📊 **Step 3/3**: Generating result visualizations...")
                progress.progress(100)

                status.update(label="✅ Analysis complete!", state="complete")

            # Store results in session state
            st.session_state["ocr_result"] = ocr_result
            st.session_state["prediction"] = prediction
            st.session_state["image"] = image
            st.session_state["mode_label"] = mode_label
            st.session_state["ocr_time"] = ocr_time
            st.session_state["pred_time"] = pred_time

        # ---- Display Results ----
        if "prediction" in st.session_state:
            prediction = st.session_state["prediction"]
            ocr_result = st.session_state["ocr_result"]
            image = st.session_state["image"]
            mode_label = st.session_state.get("mode_label", "")
            ocr_time = st.session_state.get("ocr_time", 0)
            pred_time = st.session_state.get("pred_time", 0)

            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

            # ---- VERDICT ----
            is_fake = prediction["predicted_class"] == 1
            verdict_class = "result-fake" if is_fake else "result-real"
            label_class = "result-label-fake" if is_fake else "result-label-real"
            icon = "🚨" if is_fake else "✅"

            st.markdown(f"""
            <div class="{verdict_class}">
                <div class="result-label {label_class}">
                    {icon} {prediction["prediction"]}
                </div>
                <div class="confidence-text">
                    Confidence: {prediction["confidence"]:.1%} &nbsp;|&nbsp;
                    Mode: {mode_label} &nbsp;|&nbsp;
                    Total time: {ocr_time + pred_time:.1f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("", unsafe_allow_html=True)

            # ---- Probability Bar ----
            prob_bar = create_probability_bar(
                prediction["probabilities"]["real"],
                prediction["probabilities"]["fake"],
            )
            st.plotly_chart(prob_bar, use_container_width=True, config={"displayModeBar": False})

            # ---- Details Columns ----
            col_left, col_right = st.columns(2)

            with col_left:
                # Confidence Gauges
                st.markdown('<div class="section-header">📊 Confidence Analysis</div>', unsafe_allow_html=True)
                gauge = create_confidence_gauge(
                    prediction["probabilities"]["real"],
                    prediction["probabilities"]["fake"],
                    prediction["prediction"],
                )
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

                # OCR Confidence
                if ocr_result["segments"]:
                    ocr_chart = create_ocr_confidence_chart(ocr_result["segments"])
                    if ocr_chart:
                        st.plotly_chart(ocr_chart, use_container_width=True, config={"displayModeBar": False})

            with col_right:
                # Extracted Text
                st.markdown('<div class="section-header">📝 Extracted Text (OCR)</div>', unsafe_allow_html=True)

                extracted_text = ocr_result["text"]
                if extracted_text:
                    st.markdown(f"""
                    <div class="extracted-text-box">
                        {extracted_text}
                    </div>
                    <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <span class="metric-tag metric-tag-blue">
                            📊 Avg OCR Confidence: {ocr_result["avg_confidence"]:.1%}
                        </span>
                        <span class="metric-tag metric-tag-purple">
                            🔤 {len(extracted_text.split())} words
                        </span>
                        <span class="metric-tag metric-tag-green">
                            📦 {len(ocr_result["segments"])} segments
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No text could be extracted from this image. The prediction is based on visual features only.")

                # Text with boxes on image
                st.markdown('<div class="section-header">🖼️ Detected Text Regions</div>', unsafe_allow_html=True)

                if ocr_result["raw_results"]:
                    annotated_img = draw_ocr_boxes(image, ocr_result["raw_results"])
                    st.image(annotated_img, caption="OCR Text Detection Boxes", use_container_width=True)
                else:
                    st.info("No text regions detected in the image.")

            # ---- Detailed Breakdown ----
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Detailed Analysis Breakdown</div>', unsafe_allow_html=True)

            detail_cols = st.columns(4)

            with detail_cols[0]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem;">📝</div>
                    <div style="color: #ccd6f6; font-weight: 600; margin: 0.3rem 0;">Text Analysis</div>
                    <div style="color: #8892b0; font-size: 0.85rem;">
                        {len(extracted_text.split()) if extracted_text else 0} words extracted<br>
                        OCR Confidence: {ocr_result["avg_confidence"]:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with detail_cols[1]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem;">🖼️</div>
                    <div style="color: #ccd6f6; font-weight: 600; margin: 0.3rem 0;">Image Analysis</div>
                    <div style="color: #8892b0; font-size: 0.85rem;">
                        {image.size[0]}×{image.size[1]} pixels<br>
                        ViT patch encoding
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with detail_cols[2]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem;">🔗</div>
                    <div style="color: #ccd6f6; font-weight: 600; margin: 0.3rem 0;">Fusion</div>
                    <div style="color: #8892b0; font-size: 0.85rem;">
                        Cross-modal attention<br>
                        Gated fusion applied
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with detail_cols[3]:
                verdict_color = "#F44336" if is_fake else "#4CAF50"
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="color: {verdict_color}; font-weight: 600; margin: 0.3rem 0;">
                        {prediction["prediction"]}
                    </div>
                    <div style="color: #8892b0; font-size: 0.85rem;">
                        Confidence: {prediction["confidence"]:.1%}<br>
                        Threshold: {threshold}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Raw Probabilities Table ----
            with st.expander("🔢 Raw Probabilities & Metadata"):
                meta_col1, meta_col2 = st.columns(2)
                with meta_col1:
                    st.json({
                        "prediction": prediction["prediction"],
                        "confidence": round(prediction["confidence"], 4),
                        "probabilities": {
                            "real": round(prediction["probabilities"]["real"], 4),
                            "fake": round(prediction["probabilities"]["fake"], 4),
                        },
                        "threshold": threshold,
                        "mode": mode_label,
                    })
                with meta_col2:
                    st.json({
                        "ocr_segments": len(ocr_result["segments"]),
                        "ocr_avg_confidence": round(ocr_result["avg_confidence"], 4),
                        "extracted_word_count": len(extracted_text.split()) if extracted_text else 0,
                        "image_size": f"{image.size[0]}x{image.size[1]}",
                        "ocr_time_seconds": round(ocr_time, 3),
                        "prediction_time_seconds": round(pred_time, 3),
                    })

    else:
        # ---- Empty State ----
        with col_result:
            st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 1.5rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔍</div>
                <div style="color: #ccd6f6; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
                    Upload an image to get started
                </div>
                <div style="color: #8892b0; font-size: 0.9rem; line-height: 1.6;">
                    The system will automatically:<br>
                    1. Extract text from the image using OCR<br>
                    2. Analyze both text and visual content<br>
                    3. Apply cross-modal attention fusion<br>
                    4. Show the verification result with confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 0.78rem; padding: 0.5rem 0 1rem 0;">
        Multimodal Transformer Framework for Fake News Detection &nbsp;|&nbsp;
        BERT + ViT + Cross-Modal Attention &nbsp;|&nbsp;
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
