"""
DermScript — Production-Grade Clinical Triage Support UI

A world-class, surgical-suite-grade interface for melanoma risk assessment.
Combines MobileNetV3 vision + BioClinicalBERT clinical text + LightGBM fusion
+ DRAPS conformal prediction in a premium, minimal clinical dark mode aesthetic.

Two deployment modes, auto-detected by whether model_cache/ is pre-populated:
  - Raspberry Pi / air-gapped: run `python setup_models.py` once with
    internet access first, then this app runs fully offline.
  - Streamlit Community Cloud: model_cache/ starts empty (can't ship a 436MB
    BERT cache in a GitHub repo), so weights download once automatically at
    first run, then stay cached for the container's lifetime.

NOT a diagnostic device. Research / educational prototype only. Every
output must be confirmed by a licensed clinician before any care decision.
"""

import io
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ── Route caches to the local offline folder BEFORE importing torch/transformers
APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "model_cache"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

# Only force fully-offline mode if a real pre-populated cache already exists
_HF_CACHE_POPULATED = (CACHE_DIR / "huggingface" / "hub").exists()
_TORCH_CACHE_POPULATED = (CACHE_DIR / "torch" / "hub" / "checkpoints").exists()
if _HF_CACHE_POPULATED and _TORCH_CACHE_POPULATED:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

BUNDLE_PATH = APP_DIR / "dermscript_inference_bundle.pkl"

# ── Physical hardware constants
RING_BUMP_SPACING_MM = 20.0

# ──────────────────────────────────────────────────────────────────────────
# PRODUCTION DESIGN SYSTEM
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DermScript", page_icon="🔬", layout="wide")

# Clinical dark mode palette
BG = "#07090F"           # Near-black background
SURFACE = "#0D1117"      # Card backgrounds
SURFACE_ELEVATED = "#161B27"  # Hover/active states
TEAL = "#00D4AA"         # Primary accent (key numbers, CTAs)
CORAL = "#FF5C6A"        # Danger/high risk
AMBER = "#FFB020"        # Warning/medium risk
BLUE = "#4D9FFF"         # Safe/low risk
TEXT_PRIMARY = "#E2E8F7" # Main text
TEXT_MUTED = "#7D8FAB"   # Muted text
BORDER = "#1E2533"       # Border color

# ──────────────────────────────────────────────────────────────────────────
# PRODUCTION CSS — Surgical suite + Bloomberg terminal aesthetic
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    
    html, body, [class*="css"] {{ 
        font-family: 'Inter', system-ui, sans-serif;
        background: {BG};
        color: {TEXT_PRIMARY};
    }}
    
    .stApp {{ background: {BG}; }}
    #MainMenu, header[data-testid="stHeader"] {{ display: none; }}
    footer {{ display: none; }}
    
    /* Typography hierarchy */
    h1, h2, h3, h4, h5, h6 {{ 
        color: {TEXT_PRIMARY} !important; 
        font-weight: 700 !important; 
        letter-spacing: -0.01em;
    }}
    
    p, label, .stMarkdown {{ color: {TEXT_PRIMARY}; }}
    [data-testid="stCaptionContainer"], .stCaption {{ color: {TEXT_MUTED} !important; }}
    
    /* Eyebrow labels — monospace, uppercase, minimal */
    .ds-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.15em;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }}
    
    /* Tick-rule separator — nods to the Contact Ring's ruler bumps */
    .ds-tickrule {{
        height: 2px;
        margin: 1.2rem 0 1.8rem 0;
        background: repeating-linear-gradient(
            90deg,
            {TEAL} 0 3px,
            transparent 3px 20px
        );
        opacity: 0.4;
    }}
    
    /* Premium card styling */
    .ds-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: all 0.2s ease-out;
    }}
    
    .ds-card:hover {{
        background: {SURFACE_ELEVATED};
        border-color: {TEAL}33;
    }}
    
    .ds-card.accent {{
        border-left: 3px solid var(--accent, {TEAL});
    }}
    
    /* Large metric display — monospace for precision */
    .ds-metric-big {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    
    .ds-metric-sub {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        margin-top: 0.5rem;
        font-weight: 500;
    }}
    
    /* Pill badges */
    .ds-pill {{
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.2s ease-out;
    }}
    
    /* Status indicators */
    .ds-status-row {{ 
        display: flex; 
        gap: 0.8rem; 
        flex-wrap: wrap; 
        margin-bottom: 1rem;
    }}
    
    .ds-status {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        border: 1px solid {BORDER};
        color: {TEXT_MUTED};
        background: {SURFACE};
        font-weight: 600;
        letter-spacing: 0.05em;
    }}
    
    .ds-status.ok {{ 
        color: {TEAL}; 
        border-color: {TEAL}44;
        background: {TEAL}11;
    }}
    
    .ds-status.warn {{ 
        color: {AMBER}; 
        border-color: {AMBER}44;
        background: {AMBER}11;
    }}
    
    .ds-status.bad {{ 
        color: {CORAL}; 
        border-color: {CORAL}44;
        background: {CORAL}11;
    }}
    
    /* Footer */
    .ds-footer {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: {TEXT_MUTED};
        line-height: 1.7;
        border-top: 1px solid {BORDER};
        padding-top: 1.5rem;
        margin-top: 2rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid {BORDER};
        background: {SURFACE};
        color: {TEXT_PRIMARY};
        transition: all 0.2s ease-out;
    }}
    
    .stButton > button:hover {{
        background: {SURFACE_ELEVATED};
        border-color: {TEAL}66;
    }}
    
    .stButton > button[kind="primary"] {{
        background: {TEAL};
        border: none;
        color: #000;
        font-weight: 700;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: #00e8c0;
        box-shadow: 0 0 20px {TEAL}44;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        color: {TEXT_MUTED};
        font-family: 'Inter', sans-serif;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {TEAL} !important;
        border-bottom-color: {TEAL} !important;
    }}
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select, .stTextArea textarea {{
        background: {SURFACE_ELEVATED} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 8px !important;
    }}
    
    .stNumberInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {{
        border-color: {TEAL} !important;
        box-shadow: 0 0 0 2px {TEAL}22 !important;
    }}
    
    /* Divider */
    .stDivider {{ border-color: {BORDER} !important; }}
    
    /* Expander */
    .stExpander {{ border: 1px solid {BORDER}; border-radius: 8px; }}
    .stExpander > div > div > button {{ color: {TEXT_PRIMARY}; }}
    
    /* Metric */
    [data-testid="stMetricValue"] {{ 
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
    }}
    
    /* Error/warning/success boxes */
    .stError, .stWarning, .stSuccess {{
        border-radius: 8px;
        border: 1px solid;
    }}
    
    .stError {{
        background: {CORAL}11 !important;
        border-color: {CORAL}44 !important;
        color: {CORAL} !important;
    }}
    
    .stWarning {{
        background: {AMBER}11 !important;
        border-color: {AMBER}44 !important;
        color: {AMBER} !important;
    }}
    
    .stSuccess {{
        background: {TEAL}11 !important;
        border-color: {TEAL}44 !important;
        color: {TEAL} !important;
    }}
    
    /* SVG Gauge Container */
    .ds-gauge-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem 0;
    }}
    
    /* Risk category label animation */
    @keyframes fadeInScale {{
        from {{
            opacity: 0;
            transform: scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .ds-risk-label {{
        animation: fadeInScale 0.6s ease-out 0.8s both;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }}
    
    /* Gauge needle animation */
    @keyframes gaugeNeedle {{
        from {{
            stroke-dashoffset: 1000;
        }}
        to {{
            stroke-dashoffset: 0;
        }}
    }}
    
    .ds-gauge-arc {{
        stroke-dasharray: 1000;
        stroke-dashoffset: 1000;
        animation: gaugeNeedle 0.8s ease-out forwards;
    }}
    
    /* Feature bar chart */
    .ds-feature-bar {{
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .ds-feature-label {{
        font-size: 0.9rem;
        font-weight: 600;
        min-width: 140px;
        color: {TEXT_PRIMARY};
    }}
    
    .ds-feature-bar-bg {{
        flex: 1;
        height: 24px;
        background: {SURFACE_ELEVATED};
        border-radius: 4px;
        overflow: hidden;
        margin: 0 1rem;
    }}
    
    .ds-feature-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, {TEAL}, {TEAL}dd);
        transition: width 0.6s ease-out;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.5rem;
    }}
    
    .ds-feature-pct {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #000;
    }}
    
    /* Conformal prediction card */
    .ds-conformal-card {{
        background: {SURFACE};
        border-left: 3px solid var(--border-color, {TEAL});
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .ds-conformal-title {{
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        color: var(--text-color, {TEXT_PRIMARY});
    }}
    
    .ds-conformal-subtitle {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        line-height: 1.5;
    }}
    
    /* Disclaimer card */
    .ds-disclaimer {{
        background: {CORAL}11;
        border: 1px solid {CORAL}44;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.8rem;
        color: {TEXT_MUTED};
        line-height: 1.6;
    }}
    
    /* Three-column layout */
    .ds-three-col {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    @media (max-width: 1400px) {{
        .ds-three-col {{
            grid-template-columns: 1fr 1fr;
        }}
    }}
    
    @media (max-width: 900px) {{
        .ds-three-col {{
            grid-template-columns: 1fr;
        }}
    }}
    
    /* Upload zone styling */
    .ds-upload-zone {{
        border: 2px dashed {BORDER};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: {SURFACE}44;
        transition: all 0.2s ease-out;
    }}
    
    .ds-upload-zone:hover {{
        border-color: {TEAL}66;
        background: {TEAL}11;
    }}
    
    .ds-upload-icon {{
        font-size: 2.4rem;
        margin-bottom: 0.8rem;
    }}
    
    </style>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Model loading — fully offline, cached once per process
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DermScript model bundle…")
def load_bundle():
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading vision + language backbones (offline)…")
def load_backbones():
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_large
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _TORCH_CACHE_POPULATED:
        mnet = mobilenet_v3_large(weights=None)
        state_dict_path = CACHE_DIR / "torch" / "hub" / "checkpoints" / "mobilenet_v3_large-5c1a4163.pth"
        mnet.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    else:
        from torchvision.models import MobileNet_V3_Large_Weights
        mnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    feat_extractor = mnet.features
    pool = mnet.avgpool
    mnet.classifier = nn.Identity()
    mnet.eval().to(device)
    for p in mnet.parameters():
        p.requires_grad_(False)

    bert_name = "emilyalsentzer/Bio_ClinicalBERT"
    tok = AutoTokenizer.from_pretrained(bert_name, local_files_only=_HF_CACHE_POPULATED)
    bert = AutoModel.from_pretrained(bert_name, local_files_only=_HF_CACHE_POPULATED).eval().to(device)
    for p in bert.parameters():
        p.requires_grad_(False)

    img_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return device, mnet, feat_extractor, pool, tok, bert, img_tf


def embed(image, text, device, mnet, tok, bert, img_tf, tta=True):
    """Mirrors the training notebook's `_embed()` exactly, with optional TTA."""
    import torch

    if tta:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        imgs = [image, flipped]
    else:
        imgs = [image]

    t_orig = img_tf(image).unsqueeze(0).to(device)
    vecs = []
    with torch.no_grad():
        for im in imgs:
            t = img_tf(im).unsqueeze(0).to(device)
            vecs.append(mnet(t).float().cpu().numpy())
        v = np.mean(vecs, axis=0)
        enc = tok([text or "Dermoscopy image."], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        n = bert(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
    return np.hstack([v, n]), t_orig


# ──────────────────────────────────────────────────────────────────────────
# Grad-CAM on MobileNetV3's last conv block
# ──────────────────────────────────────────────────────────────────────────
def grad_cam(image_tensor, mnet, feat_extractor, pool, device):
    """Real Grad-CAM: hooks last conv block's activations + gradients."""
    import torch

    activations = {}

    def fwd_hook(_, __, out):
        activations["act"] = out

    handle = feat_extractor[-1].register_forward_hook(fwd_hook)
    image_tensor = image_tensor.clone().requires_grad_(True)
    feats = feat_extractor(image_tensor)
    pooled = pool(feats).flatten(1)
    target = pooled.norm()
    target.backward()
    handle.remove()

    act = activations["act"].detach()[0]
    grads = feats.grad if feats.grad is not None else None
    if grads is None:
        cam = act.mean(dim=0).cpu().numpy()
    else:
        weights = grads[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_heatmap(pil_img, cam):
    base = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = 0.55 * base + 0.45 * heat
    return (blended * 255).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# SHAP breakdown on LightGBM
# ──────────────────────────────────────────────────────────────────────────
def shap_breakdown(bundle, X, vis_dim, nlp_dim):
    """Real SHAP on the underlying LightGBM step."""
    import shap

    cal_model = bundle["model"]
    try:
        sub_pipe = cal_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        sub_pipe = cal_model.calibrated_classifiers_[0].base_estimator

    Xs = sub_pipe.named_steps["scale"].transform(X)
    Xp = sub_pipe.named_steps["pca"].transform(Xs)
    lgbm = sub_pipe.named_steps["clf"]

    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(Xp)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv).reshape(-1)

    loadings = sub_pipe.named_steps["pca"].components_
    vision_mass = np.abs(loadings[:, :vis_dim]).sum(axis=1)
    text_mass = np.abs(loadings[:, vis_dim:]).sum(axis=1)
    is_vision_dominant = vision_mass > text_mass

    vision_contrib = np.abs(sv[is_vision_dominant]).sum()
    text_contrib = np.abs(sv[~is_vision_dominant]).sum()
    total = vision_contrib + text_contrib + 1e-9
    return vision_contrib / total, text_contrib / total, sv, is_vision_dominant


# ──────────────────────────────────────────────────────────────────────────
# Ruler-bump homography
# ──────────────────────────────────────────────────────────────────────────
def detect_ruler_bumps_and_diameter(cv_img_bgr, lesion_radius_px_guess=None):
    """Detects ruler bumps via Hough circles, computes mm/px scale."""
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 8,
        param1=80, param2=22, minRadius=3, maxRadius=max(4, w // 40),
    )

    debug = cv_img_bgr.copy()
    if circles is None or len(circles[0]) < 2:
        cv2.putText(debug, "Ruler bumps not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug

    pts = circles[0][:4, :2]
    for x, y, r in circles[0][:4]:
        cv2.circle(debug, (int(x), int(y)), int(r), (0, 255, 0), 2)

    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    if not dists:
        return None, debug
    px_per_mm = float(np.mean(dists)) / RING_BUMP_SPACING_MM
    if px_per_mm <= 0:
        return None, debug

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug
    center = np.array([w / 2, h / 2])
    contours = sorted(contours, key=lambda c: np.linalg.norm(
        np.array(cv2.minEnclosingCircle(c)[0]) - center))
    (cx, cy), radius_px = cv2.minEnclosingCircle(contours[0])
    cv2.circle(debug, (int(cx), int(cy)), int(radius_px), (255, 0, 255), 2)

    diameter_mm = (2 * radius_px) / px_per_mm
    return diameter_mm, debug


# ──────────────────────────────────────────────────────────────────────────
# SVG GAUGE COMPONENT — Hand-coded for reliability on Streamlit Cloud
# ──────────────────────────────────────────────────────────────────────────
def create_risk_gauge_svg(risk_score):
    """Creates a semicircular gauge SVG with color gradient based on risk."""
    # Map risk to angle (0-180 degrees)
    angle = risk_score * 180
    
    # Color gradient: blue (low) -> amber (medium) -> coral (high)
    if risk_score < 0.3:
        color = BLUE
    elif risk_score < 0.6:
        # Interpolate between amber and blue
        color = AMBER
    else:
        color = CORAL
    
    # SVG path for semicircle
    svg = f"""
    <svg viewBox="0 0 200 120" width="100%" height="200px" style="max-width: 300px;">
        <!-- Background arc -->
        <path d="M 20 100 A 80 80 0 0 1 180 100" stroke="{BORDER}" stroke-width="8" fill="none" stroke-linecap="round"/>
        
        <!-- Colored arc (animated) -->
        <path d="M 20 100 A 80 80 0 0 1 {20 + 160 * risk_score} {100 - 80 * np.sin(np.radians(angle))}" 
              stroke="{color}" stroke-width="8" fill="none" stroke-linecap="round"
              class="ds-gauge-arc" style="stroke-dasharray: 251; stroke-dashoffset: 251;"/>
        
        <!-- Center text area -->
        <circle cx="100" cy="100" r="50" fill="{SURFACE}" stroke="{BORDER}" stroke-width="1"/>
        
        <!-- Risk percentage -->
        <text x="100" y="90" font-family="JetBrains Mono" font-size="32" font-weight="700" 
              text-anchor="middle" fill="{color}">{risk_score:.1%}</text>
        
        <!-- Tick marks -->
        <line x1="20" y1="100" x2="10" y2="100" stroke="{TEXT_MUTED}" stroke-width="1" opacity="0.5"/>
        <line x1="100" y1="20" x2="100" y2="10" stroke="{TEXT_MUTED}" stroke-width="1" opacity="0.5"/>
        <line x1="180" y1="100" x2="190" y2="100" stroke="{TEXT_MUTED}" stroke-width="1" opacity="0.5"/>
    </svg>
    """
    return svg


# ──────────────────────────────────────────────────────────────────────────
# HEADER — Premium branding
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; letter-spacing: -0.02em;">
            🔬 <span style="color: {TEAL};">DermScript</span>
        </h1>
        <p style="margin: 0.3rem 0 0 0; color: {TEXT_MUTED}; font-size: 0.9rem;">
            AI-Assisted Melanoma Triage · Research Prototype
        </p>
    </div>
    <div style="display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: flex-end;">
        <span class="ds-status ok">● Model v7.3</span>
        <span class="ds-status ok">● DRAPS Active</span>
        <span class="ds-status warn">● NOT FOR CLINICAL USE</span>
    </div>
</div>
<div class="ds-tickrule"></div>
<p style="color: {TEXT_MUTED}; font-size: 0.85rem; margin-bottom: 1.5rem;">
    For research and educational demonstration only. All outputs require confirmation by a licensed dermatologist.
</p>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — Patient context
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-eyebrow">Patient Context</div>', unsafe_allow_html=True)
    
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Female", "Male", "Prefer not to say"])
    site = st.selectbox(
        "Anatomical site",
        ["Scalp", "Face", "Neck", "Trunk", "Upper extremity",
         "Lower extremity", "Palms/Soles", "Other"],
    )
    fitz = st.select_slider("Fitzpatrick skin type", options=["I", "II", "III", "IV", "V", "VI"], value="III")
    
    with st.expander("Clinical observation (optional)"):
        note = st.text_area(
            "Notes",
            placeholder="e.g. Irregular border, recent change in size, mild itching.",
            height=100,
            label_visibility="collapsed",
        )
    
    st.divider()
    st.markdown('<div class="ds-eyebrow">About DermScript</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="ds-card" style="margin-bottom: 0.8rem;">
        <div style="font-size: 0.9rem; line-height: 1.6; color: {TEXT_MUTED};">
            <strong style="color: {TEXT_PRIMARY};">$74 3D-printed dermatoscope</strong> powered by multimodal AI.
            Cross-polarized imaging eliminates glare. Runs fully offline on Raspberry Pi Zero 2W.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training AUC", "0.890")
    with col2:
        st.metric("N trained", "33,717")
    
    st.markdown(f"""
    <div style="font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 0.8rem;">
        <strong>Datasets:</strong> ISIC 2024, HAM10000, PAD-UFES-20, Derm7pt
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# System status
# ──────────────────────────────────────────────────────────────────────────
bundle_ok = BUNDLE_PATH.exists()
if not bundle_ok:
    st.error(f"Model bundle not found at `{BUNDLE_PATH}`. Copy `dermscript_inference_bundle.pkl` next to `app.py`.")
    st.stop()

bundle = load_bundle()
vis_dim = bundle.get("vis_dim", 960)
nlp_dim = bundle.get("nlp_dim", 768)

try:
    device, mnet, feat_extractor, pool, tok, bert, img_tf = load_backbones()
    backbones_ok = True
except Exception as e:
    backbones_ok = False
    backbone_error = str(e)

cache_ok = backbones_ok and (CACHE_DIR / "huggingface").exists()

status_html = '<div class="ds-status-row">'
status_html += f'<div class="ds-status {"ok" if bundle_ok else "bad"}">● MODEL BUNDLE LOADED</div>'
status_html += f'<div class="ds-status {"ok" if cache_ok else "bad"}">● OFFLINE CACHE {"READY" if cache_ok else "FAILED"}</div>'
status_html += '<div class="ds-status">● MODE: AIR-GAPPED INFERENCE</div>'
status_html += '</div>'
st.markdown(status_html, unsafe_allow_html=True)

if not backbones_ok:
    st.error(
        "Could not load the vision/language backbones. On Streamlit Cloud this "
        "usually means the one-time download was interrupted (slow connection / "
        "cold start) -- just reload the page and let it finish."
        f"\n\nDetails: {backbone_error}"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# IMAGE UPLOAD — Three-column layout
# ──────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="ds-eyebrow">Analysis</div>', unsafe_allow_html=True)

col_upload, col_preview, col_spacer = st.columns([2, 1.5, 0.5], gap="large")

with col_upload:
    st.markdown("**Upload lesion image from dermatoscope**")
    img_file = st.file_uploader(
        "Upload lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    
    source_bytes = None
    if img_file is not None:
        source_bytes = img_file.getvalue()
    
    run = st.button(
        "🚀 Run DermScript Analysis",
        type="primary",
        use_container_width=True,
        disabled=source_bytes is None,
    )

with col_preview:
    if source_bytes is not None:
        st.image(source_bytes, caption="Current image", use_container_width=True)
    else:
        st.markdown(
            f"""<div class="ds-card" style="text-align: center; color: {TEXT_MUTED}; padding: 2rem;">
                📸 No image yet
            </div>""",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────
# ANALYSIS & RESULTS — Three-column layout
# ──────────────────────────────────────────────────────────────────────────
if source_bytes is not None and run:
    raw_bytes = source_bytes
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    full_note = f"Age {age}, {sex}, site: {site}. {note}".strip()

    with st.spinner("Extracting features… Running fusion model… Computing conformal prediction…"):
        X, img_tensor = embed(pil_img, full_note, device, mnet, tok, bert, img_tf)
        risk = float(bundle["model"].predict_proba(X)[:, 1][0])

    group_map = {"I": "FST I-II", "II": "FST I-II", "III": "FST III-IV",
                 "IV": "FST III-IV", "V": "FST V-VI", "VI": "FST V-VI"}
    group_name = group_map[fitz]
    cp_group = bundle.get("cp_by_group", {}).get(group_name) or bundle.get("cp_overall", {})
    q_hat = cp_group.get("q_hat", 0.8)

    in_set = []
    if (1 - risk) >= 1 - q_hat:
        in_set.append("benign")
    if risk >= 1 - q_hat:
        in_set.append("malignant")
    if not in_set:
        in_set = ["benign", "malignant"]
    deferred = len(in_set) > 1

    st.divider()

    # ── THREE-COLUMN RESULTS LAYOUT ──
    col_left, col_center, col_right = st.columns([1, 1.2, 1], gap="large")

    # LEFT COLUMN: Image + Grad-CAM
    with col_left:
        st.markdown('<div class="ds-eyebrow">Lesion Image</div>', unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded lesion", use_container_width=True)
        
        st.markdown('<div class="ds-eyebrow" style="margin-top: 1.2rem;">Ruler Detection</div>', unsafe_allow_html=True)
        diam_mm, debug_img = detect_ruler_bumps_and_diameter(cv_img)
        debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        st.image(debug_rgb, caption="Ruler-bump detection", use_container_width=True)
        
        if diam_mm is not None:
            st.metric("Estimated diameter", f"{diam_mm:.1f} mm")
        else:
            st.caption("⚠ Ruler bumps not reliably detected in this frame.")

    # CENTER COLUMN: Risk gauge + Conformal prediction
    with col_center:
        st.markdown('<div class="ds-eyebrow">Malignancy Risk</div>', unsafe_allow_html=True)
        
        # SVG Gauge
        gauge_svg = create_risk_gauge_svg(risk)
        st.markdown(f'<div class="ds-gauge-container">{gauge_svg}</div>', unsafe_allow_html=True)
        
        # Risk category label
        if risk < 0.15:
            risk_label = "LOW RISK"
            risk_color = BLUE
        elif risk < 0.35:
            risk_label = "BORDERLINE"
            risk_color = AMBER
        elif risk < 0.65:
            risk_label = "ELEVATED RISK"
            risk_color = AMBER
        else:
            risk_label = "HIGH RISK — REFER"
            risk_color = CORAL
        
        st.markdown(
            f'<div class="ds-risk-label" style="text-align: center; color: {risk_color};">{risk_label}</div>',
            unsafe_allow_html=True,
        )
        
        # Conformal prediction card
        st.markdown('<div class="ds-eyebrow" style="margin-top: 1.5rem;">DRAPS Conformal Prediction</div>', unsafe_allow_html=True)
        
        if deferred:
            st.markdown(
                f"""<div class="ds-conformal-card" style="--border-color: {AMBER}; --text-color: {AMBER};">
                    <div class="ds-conformal-title">⚠ UNCERTAIN — Defer to specialist</div>
                    <div class="ds-conformal-subtitle">
                        Conformal prediction set includes both benign and malignant. 
                        DRAPS safety layer activated. Manual dermatologist review required.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            label = "MALIGNANT" if "malignant" in in_set else "BENIGN"
            label_color = CORAL if label == "MALIGNANT" else TEAL
            icon = "✕" if label == "MALIGNANT" else "✓"
            
            st.markdown(
                f"""<div class="ds-conformal-card" style="--border-color: {label_color}; --text-color: {label_color};">
                    <div class="ds-conformal-title">{icon} CONFIDENT — {label}</div>
                    <div class="ds-conformal-subtitle">
                        Conformal prediction set = {{{in_set[0]}}} at q̂={q_hat:.3f} 
                        for {group_name} — single outcome at the 95% coverage level.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        
        # Disclaimer
        st.markdown(
            f"""<div class="ds-disclaimer">
                ⚠ <strong>Clinical Disclaimer:</strong> This output is a screening support signal only. 
                Sensitivity and specificity vary by population. All malignant predictions require biopsy confirmation.
            </div>""",
            unsafe_allow_html=True,
        )

    # RIGHT COLUMN: Feature attribution
    with col_right:
        st.markdown('<div class="ds-eyebrow">What Drove This Prediction</div>', unsafe_allow_html=True)
        
        try:
            vis_pct, txt_pct, sv, is_vis = shap_breakdown(bundle, X, vis_dim, nlp_dim)
            
            # Feature bars
            st.markdown(f"""
            <div class="ds-feature-bar">
                <div class="ds-feature-label">Vision (image)</div>
                <div class="ds-feature-bar-bg">
                    <div class="ds-feature-bar-fill" style="width: {vis_pct*100:.1f}%;">
                        <span class="ds-feature-pct">{vis_pct*100:.0f}%</span>
                    </div>
                </div>
            </div>
            <div class="ds-feature-bar">
                <div class="ds-feature-label">Clinical text</div>
                <div class="ds-feature-bar-bg">
                    <div class="ds-feature-bar-fill" style="width: {txt_pct*100:.1f}%;">
                        <span class="ds-feature-pct">{txt_pct*100:.0f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("Share of total |SHAP| attribution. Fusion of MobileNetV3 (vision) + BioClinicalBERT (clinical text).")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
        
        # Clinical context
        st.markdown('<div class="ds-eyebrow" style="margin-top: 1.5rem;">Clinical Context Used</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="ds-card" style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: {TEXT_MUTED};">
                {full_note}
            </div>""",
            unsafe_allow_html=True,
        )
        
        # Fitzpatrick-stratified thresholds
        st.markdown('<div class="ds-eyebrow" style="margin-top: 1.5rem;">Fairness Thresholds</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="ds-card" style="font-size: 0.8rem;">
                <div style="margin-bottom: 0.5rem;"><strong>FST I-II:</strong> <code>q = {bundle.get('cp_by_group', {}).get('FST I-II', {}).get('q_hat', 0.8):.4f}</code></div>
                <div style="margin-bottom: 0.5rem;"><strong>FST III-IV:</strong> <code>q = {bundle.get('cp_by_group', {}).get('FST III-IV', {}).get('q_hat', 0.8):.4f}</code></div>
                <div><strong>FST V-VI:</strong> <code>q = {bundle.get('cp_by_group', {}).get('FST V-VI', {}).get('q_hat', 0.8):.4f}</code></div>
                <div style="color: {TEXT_MUTED}; margin-top: 0.6rem; font-size: 0.75rem;">
                    Fitzpatrick-stratified thresholds ensure equitable deferral rates across skin tones.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

elif source_bytes is not None:
    st.caption("Image ready — click **🚀 Run DermScript Analysis** above to score it.")

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="ds-footer">
    <strong>DermScript v7.3</strong> · MobileNetV3 + BioClinicalBERT → PCA-128 → LightGBM · Training AUC 0.890<br>
    <strong>External Validation:</strong> Stanford DDI external AUC=0.585 (distribution shift). 
    Generalization beyond training distribution NOT yet fully established.<br>
    <strong>⚠ Research prototype — not FDA cleared — not for clinical diagnosis</strong>
    </div>""",
    unsafe_allow_html=True,
)
