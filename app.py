import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="AI Eye Gender Classification",
    page_icon="👁️",
    layout="wide"
)

# ------------------------------------
# Custom CSS
# ------------------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#2E86DE;
}

.sub-title{
    text-align:center;
    font-size:20px;
    color:gray;
    margin-bottom:25px;
}

.prediction-card{
    background:#F4F6F7;
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 0px 10px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Load Model
# ------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

IMG_SIZE = 128

# ------------------------------------
# Header
# ------------------------------------
st.markdown('<p class="main-title">👁️ AI Eye Gender Classification</p>', unsafe_allow_html=True)

st.markdown(
'<p class="sub-title">Upload an eye image and let the AI predict whether it belongs to a Male or Female.</p>',
unsafe_allow_html=True)

st.divider()

# ------------------------------------
# Upload
# ------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload an Eye Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img=image.resize((IMG_SIZE,IMG_SIZE))
    img=np.array(img)/255.0
    img=np.expand_dims(img,0)

    with st.spinner("Analyzing Image..."):
        pred=model.predict(img,verbose=0)

    confidence=float(pred[0][0])

    if confidence>0.5:
        label="👨 Male"
        score=confidence
    else:
        label="👩 Female"
        score=1-confidence

    with col2:

        st.markdown('<div class="prediction-card">',unsafe_allow_html=True)

        st.markdown("## ✅ Prediction")

        st.success(label)

        st.metric("Confidence",f"{score*100:.2f}%")

        st.progress(int(score*100))

        st.markdown("</div>",unsafe_allow_html=True)

    st.balloons()

st.divider()

with st.expander("📖 How to Use"):
    st.write("""
1. Upload a clear eye image.
2. Wait a few seconds.
3. View the AI prediction and confidence score.
""")

st.caption("Developed using Deep Learning & Streamlit")
