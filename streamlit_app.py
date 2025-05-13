import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

# ---- CUSTOM CSS FOR BACKGROUND AND FONTS ----
st.markdown("""
    <style>
    body {
        background-image: linear-gradient(to bottom right, #e0f7fa, #f1f8e9);
        font-size: 18px;
    }
    .main h1 {
        font-size: 48px !important;
    }
    .main p, .main label, .main div, .main button, .main span {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🧠 Brain Tumor Classifier</h1>
    <p style='text-align: center;'>Upload a brain MRI image to classify tumor type using a trained DenseNet201 model.</p>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Options")
    st.write("This demo uses a pre-trained DenseNet201 model.")
    st.markdown("---")
    st.markdown("Model: **DenseNet201**")
    st.markdown("Classes: `glioma`, `meningioma`, `no_tumor`, `pituitary`")
    st.markdown("---")
    st.write("📞 Contact: brainai@appsupport.com")
    st.write("🔗 [GitHub Repo](https://github.com/SouMahdi/brain_tumor_classifier)")

# ---- LOAD MODEL ----
model = load_model("brain_tumor_classifier.h5")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ---- FILE UPLOADER ----
st.markdown("### 📤 Upload an MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🔍 Classification Result")

        # Preprocess image
        resized = image.resize((224, 224))
        image_array = img_to_array(resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        if st.button("🚀 Classify Tumor"):
            predictions = model.predict(image_array)[0]
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)

            st.success(f"🧠 Predicted Tumor Type: `{predicted_class}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}`")

            with st.expander("📊 Show Raw Prediction Scores"):
                for i, score in enumerate(predictions):
                    st.write(f"{class_names[i]}: {score:.2f}")

else:
    st.info("Please upload an image to get started.")
