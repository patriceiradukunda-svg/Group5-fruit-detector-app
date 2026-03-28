Fix it:
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from utils import (
    load_model,
    predict_and_annotate,
    build_summary_text,
    format_detection_table,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Group 5 Fruit Detector",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top, #fff6cf 0%, #ffe3f0 18%, #f1e5ff 38%, #dbf3ff 62%, #e6ffe9 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff7ed 0%, #fdf2f8 45%, #ecfeff 100%);
        border-right: 1px solid rgba(255,255,255,0.85);
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #c026d3, #f97316, #65a30d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-box {
        position: relative;
        overflow: hidden;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .soft-card {
        background: rgba(255,255,255,0.84);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.85);
    }

    .metric-card {
        background: rgba(255,255,255,0.88);
        border-radius: 20px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }

    .small-label {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 0.15rem;
    }

    .big-number {
        font-size: 2rem;
        font-weight: 800;
        color: #f97316;
    }

    .pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        margin: 4px 6px 4px 0;
        font-size: 0.95rem;
    }

    .apple { background: linear-gradient(90deg, #fb7185, #ef4444); }
    .kiwi { background: linear-gradient(90deg, #84cc16, #16a34a); }
    .orange { background: linear-gradient(90deg, #fdba74, #f97316); }
    .pear { background: linear-gradient(90deg, #6ee7b7, #a3e635); }
    .strawberry { background: linear-gradient(90deg, #f472b6, #f43f5e); }
    .tomato { background: linear-gradient(90deg, #f87171, #fb923c); }

    .info-box {
        background: linear-gradient(135deg, #fff7ed, #fdf2f8, #ecfeff);
        border-radius: 22px;
        padding: 16px;
        color: #334155;
        border: 1px solid #ffffff;
    }

    .sidebar-box {
        background: rgba(255,255,255,0.75);
        padding: 14px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.9);
        margin-bottom: 12px;
    }

    .team-card {
        background: rgba(255,255,255,0.85);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        min-height: 150px;
    }

    .floating-wrap {
        position: absolute;
        inset: 0;
        pointer-events: none;
        overflow: hidden;
    }

    .float-emoji {
        position: absolute;
        font-size: 2rem;
        opacity: 0.75;
        animation: floaty 7s ease-in-out infinite;
    }

    .emoji1 { top: 12%; left: 6%; animation-delay: 0s; }
    .emoji2 { top: 18%; right: 10%; animation-delay: 1s; }
    .emoji3 { bottom: 14%; left: 18%; animation-delay: 2s; }
    .emoji4 { bottom: 18%; right: 20%; animation-delay: 3s; }
    .emoji5 { top: 50%; right: 42%; animation-delay: 4s; }

    @keyframes floaty {
        0%   { transform: translateY(0px) translateX(0px) rotate(0deg); }
        50%  { transform: translateY(-14px) translateX(8px) rotate(6deg); }
        100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        padding-top: 10px;
        padding-bottom: 30px;
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem;
        }

        .hero-box {
            padding: 18px;
            border-radius: 22px;
        }

        .soft-card, .metric-card, .team-card {
            border-radius: 18px;
            padding: 14px;
        }

        .pill {
            font-size: 0.85rem;
            padding: 7px 12px;
        }

        .float-emoji {
            font-size: 1.6rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL
# -----------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# -----------------------------
# HELPERS
# -----------------------------
SUPPORTED_CLASSES = ["apple", "kiwi", "orange", "pear", "strawberry", "tomato"]

def pil_image_to_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-box">
        <h2 style="margin:0; color:#a21caf;">🍓 Group 5</h2>
        <div style="color:#475569; margin-top:4px;">Fun Fruit Vision</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Detector", "About", "Team"],
        index=0
    )

    st.markdown("""
    <div class="sidebar-box">
        <b>Supported fruits</b><br>
        apple, kiwi, orange, pear, strawberry, tomato
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <b>Tip</b><br>
        Upload a clear fruit image for better detection.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# DETECTOR PAGE
# -----------------------------
if page == "Detector":
    st.markdown("""
    <div class="hero-box">
        <div class="floating-wrap">
            <div class="float-emoji emoji1">🍎</div>
            <div class="float-emoji emoji2">🍊</div>
            <div class="float-emoji emoji3">🍓</div>
            <div class="float-emoji emoji4">🍐</div>
            <div class="float-emoji emoji5">🥝</div>
        </div>
if page == "Detector":
    st.markdown("""
    <div class="hero-box">
        <div class="floating-wrap">
            <div class="float-emoji emoji1">🍎</div>
            <div class="float-emoji emoji2">🍊</div>
            <div class="float-emoji emoji3">🍓</div>
            <div class="float-emoji emoji4">🍐</div>
            <div class="float-emoji emoji5">🥝</div>
        </div>

        <div style="font-size:1rem; font-weight:700; color:#a21caf;">
            ✨ Group 5 Fruit Detector
        </div>
        <div class="main-title">Fun Fruit Vision</div>
        <div style="font-size:1.08rem; color:#475569; max-width:860px;">
            Upload an image and let our playful AI detect fruits inside it.
            The app shows what fruits were found, how sure the model is, and a friendly summary.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("""
        <div class="soft-card">
            <h3 style="margin-top:0; color:#0f172a;">🍓 Supported Classes</h3>
            <div style="color:#475569; margin-bottom:10px;">
                This platform is trained to detect only the following classes:
            </div>
            <span class="pill apple">apple</span>
            <span class="pill kiwi">kiwi</span>
            <span class="pill orange">orange</span>
            <span class="pill pear">pear</span>
            <span class="pill strawberry">strawberry</span>
            <span class="pill tomato">tomato</span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="metric-card">
                <div class="small-label">Fruit classes</div>
                <div class="big-number">6</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="metric-card">
                <div class="small-label">Powered by</div>
                <div class="big-number" style="color:#c026d3;">YOLO</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload a fruit image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05
        )

        detect_clicked = st.button("🔍 Detect Fruits", use_container_width=True)

        st.markdown("""
        <div class="info-box" style="margin-top:14px;">
            <b>Friendly Note:</b> This platform is trained only for apple, kiwi, orange, pear,
            strawberry, and tomato. If a different object appears in the image, it may not be detected.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Preview")

        if uploaded_file is not None:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image, use_container_width=True)
        else:
            st.info("Upload an image to preview it here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    if uploaded_file is not None and detect_clicked:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting fruits..."):
            annotated_image, detections = predict_and_annotate(
                model=model,
                image=image,
                conf_threshold=conf_threshold
            )

        summary = build_summary_text(detections)
        table_data = format_detection_table(detections)

        result_col1, result_col2 = st.columns([1.1, 0.9])

        with result_col1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("✅ Detection Result")
            st.image(annotated_image, use_container_width=True)

            image_bytes = pil_image_to_bytes(annotated_image)
            st.download_button(
                label="⬇️ Download detected image",
                data=image_bytes,
                file_name="group5_detected_fruits.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("📋 Detection Summary")
            st.success(summary)

            if len(table_data) > 0:
                st.subheader("🍎 Detected Fruits Details")
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No supported fruits were detected in this image.")
            st.markdown('</div>', unsafe_allow_html=True)

    examples_dir = Path("assets/examples")
    if examples_dir.exists():
        example_files = (
            list(examples_dir.glob("*.jpg")) +
            list(examples_dir.glob("*.png")) +
            list(examples_dir.glob("*.jpeg"))
        )
        if example_files:
            st.markdown("")
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("🌈 Example Images")
            st.caption("Place sample fruit images inside assets/examples/")

            ex_cols = st.columns(min(len(example_files), 4))
            for i, img_path in enumerate(example_files[:4]):
                with ex_cols[i]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "About":
    st.markdown("""
    <div class="hero-box">
        <div class="main-title">About Fun Fruit Vision</div>
        <div style="font-size:1.08rem; color:#475569; max-width:860px;">
            Fun Fruit Vision is a bright and friendly fruit detection platform designed by Group 5.
            It allows users to upload an image and receive fruit detection results with bounding boxes,
            confidence scores, and an easy summary.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="soft-card">
        <h3 style="margin-top:0;">🎯 What this platform does</h3>
        <ul>
            <li>Accepts uploaded fruit images</li>
            <li>Detects fruits using a trained YOLO model</li>
            <li>Shows labeled bounding boxes</li>
            <li>Provides confidence scores and a simple summary</li>
            <li>Supports computer, tablet, and mobile users</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="soft-card">
        <h3 style="margin-top:0;">🍇 Supported classes</h3>
        <p>This app is trained only on these classes:</p>
        <span class="pill apple">apple</span>
        <span class="pill kiwi">kiwi</span>
        <span class="pill orange">orange</span>
        <span class="pill pear">pear</span>
        <span class="pill strawberry">strawberry</span>
        <span class="pill tomato">tomato</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# TEAM PAGE
# -----------------------------
elif page == "Team":
    st.markdown("""
    <div class="hero-box">
        <div class="main-title">Meet Group 5</div>
        <div style="font-size:1.08rem; color:#475569; max-width:860px;">
            This section presents the team behind the fruit detection platform.
            Replace the sample names below with your real group members.
        </div>
    </div>
    """, unsafe_allow_html=True)

    team_cols = st.columns(3)

    team_members = [
        {
            "name": "Member 1",
            "role": "Project Lead",
            "desc": "Led project planning, coordination, and presentation preparation."
        },
        {
            "name": "Member 2",
            "role": "Model Training",
            "desc": "Worked on data preparation, model training, and evaluation."
        },
        {
            "name": "Member 3",
            "role": "Web App Development",
            "desc": "Designed the user interface and connected the trained model to the app."
        },
        {
            "name": "Member 4",
            "role": "Data Analysis, Testing & Documentation",
            "desc": "Handled exploratory analysis, interpretation, and reporting. Tested the platform and contributed to documentation and improvements."
        },
        {
            "name": "Lecturer / Tutors",
            "role": "Guidance",
            "desc": "Supported the team with direction, review, and feedback."
        },
    ]

    for i, member in enumerate(team_members):
        with team_cols[i % 3]:
            st.markdown(f"""
            <div class="team-card">
                <h3 style="margin-top:0; color:#c026d3;">👤 {member['name']}</h3>
                <p><b>{member['role']}</b></p>
                <p style="color:#475569;">{member['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer-note">
    Made by Group 5 💜🍊🍏
</div>
""", unsafe_allow_html=True)
