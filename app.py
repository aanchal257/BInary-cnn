import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="AI Eye Gender Classification",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------------------
# Load Model
# -----------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

IMG_SIZE = 128

# -----------------------------------------
# Custom CSS
# -----------------------------------------
st.markdown("""
<style>

.stApp{
    background:#edf4ff;
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero Banner */

.hero{
    padding:35px;
    border-radius:18px;
    background:linear-gradient(90deg,#2563EB,#4F46E5,#7C3AED);
    color:white;
    text-align:center;
    margin-bottom:30px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.2);
}

.hero h1{
    font-size:44px;
    margin-bottom:8px;
}

.hero p{
    font-size:18px;
    opacity:0.9;
}

/* Cards */

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 4px 18px rgba(0,0,0,0.12);
}

/* Upload Box */

[data-testid="stFileUploader"]{
    background:white;
    border-radius:15px;
    padding:15px;
    border:2px dashed #4F46E5;
}

/* Success Box */

div[data-testid="stSuccess"]{
    border-radius:12px;
}

/* Metric */

[data-testid="metric-container"]{
    background:#F8FAFC;
    border-radius:15px;
    padding:12px;
    border:1px solid #E2E8F0;
}

/* Progress */

.stProgress > div > div > div{
    background:#2563EB;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Banner
# -----------------------------------------

st.markdown("""
<div class="hero">
<h1>👁️ AI Eye Gender Classification</h1>
<p>Deep Learning Powered Prediction System</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Upload
# -----------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload an Eye Image",
    type=["jpg","jpeg","png"]
)

# -----------------------------------------
# Prediction
# -----------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📸 Uploaded Image")

        st.image(image, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    img = image.resize((IMG_SIZE,IMG_SIZE))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    with st.spinner("🔍 AI is analyzing the image..."):

        prediction = model.predict(img, verbose=0)

    confidence = float(prediction[0][0])

    # Change labels if your classes are reversed
    if confidence >= 0.5:
        label = "👨 Male"
        score = confidence
    else:
        label = "👩 Female"
        score = 1-confidence

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🤖 Prediction")

        st.success(label)

        st.metric(
            "Confidence Score",
            f"{score*100:.2f}%"
        )

        st.progress(score)

        st.markdown("</div>", unsafe_allow_html=True)

    st.balloons()

# -----------------------------------------
# Bottom Tip
# -----------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.info("💡 For best results, upload a clear, high-quality eye image.")
