import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Male & Female Eye Classifier",
    page_icon="👁️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

IMG_SIZE = 128

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 About Project")

st.sidebar.info(
"""
This application uses a **Convolutional Neural Network (CNN)** to classify
an uploaded eye image as **Male** or **Female**.

### Tech Stack
- Python
- TensorFlow / Keras
- CNN
- Streamlit
- NumPy
- Pillow

### Input
Upload an eye image in JPG, JPEG, or PNG format.
"""
)

# ----------------------------
# Title
# ----------------------------
st.title("👁️ Male & Female Eye Classification using CNN")

st.markdown(
"""
Upload an eye image below and let the trained CNN model predict
whether it belongs to a **Male** or **Female**.
"""
)

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img_array, verbose=0)

    confidence = float(prediction[0][0])

    # -------------------------------------------------
    # IMPORTANT
    # Change this if your classes are reversed.
    # -------------------------------------------------

    if confidence >= 0.5:
        predicted_class = "👨 Male"
        final_confidence = confidence
    else:
        predicted_class = "👩 Female"
        final_confidence = 1 - confidence

    with col2:

        st.subheader("Prediction Result")

        st.success(predicted_class)

        st.metric(
            label="Confidence",
            value=f"{final_confidence*100:.2f}%"
        )

        st.progress(int(final_confidence*100))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.markdown(
"""
### 📖 Workflow

1. Upload an eye image.
2. Image is resized to **128×128**.
3. Pixel values are normalized.
4. CNN extracts important eye features.
5. Model predicts **Male** or **Female**.
6. Confidence score is displayed.

---
Made with ❤️ using **TensorFlow** and **Streamlit**
"""
)
