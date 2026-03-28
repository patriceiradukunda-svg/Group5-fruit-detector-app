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
.main-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c026d3, #f97316, #65a30d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.soft-card {
    background: rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 20px;
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
def pil_image_to_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    page = st.radio("Navigate", ["Detector", "About", "Team"])

# -----------------------------
# DETECTOR PAGE
# -----------------------------
if page == "Detector":
    st.markdown("""
    <div class="soft-card">
        <div style="font-size:1rem; font-weight:700;">✨ Group 5 Fruit Detector</div>
        <div class="main-title">Fun Fruit Vision</div>
        <p>Upload an image and detect fruits using AI.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        conf = st.slider("Confidence", 0.1, 0.9, 0.25)
        detect = st.button("Detect")

    with col2:
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_container_width=True)

    if uploaded_file and detect:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting..."):
            annotated, detections = predict_and_annotate(
                model=model,
                image=image,
                conf_threshold=conf
            )

        st.image(annotated, use_container_width=True)

        summary = build_summary_text(detections)
        st.success(summary)

        table = format_detection_table(detections)
        if len(table) > 0:
            st.dataframe(table, use_container_width=True)

        img_bytes = pil_image_to_bytes(annotated)
        st.download_button(
            "Download Image",
            img_bytes,
            file_name="detected.png",
            mime="image/png"
        )

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "About":
    st.title("About")
    st.write("This app detects fruits using a YOLO model.")

# -----------------------------
# TEAM PAGE
# -----------------------------
elif page == "Team":
    st.title("Team")
    st.write("Group 5 Project")
