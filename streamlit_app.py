import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np

# 🔧 Load your trained model
model = load_model("brain_tumor_classifier.h5")

# 🧠 Set your tumor class names manually
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# 🎨 Title and description
st.title("🧠 Brain Tumor Classifier")
st.markdown("Upload a brain MRI image and classify it into one of four tumor types.")

# 📂 File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔄 Preprocess image
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # ▶️ Predict button
    if st.button("Classify Tumor"):
        prediction = model.predict(image_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # 🎯 Show prediction
        st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
        st.markdown(f"**Confidence:** {confidence:.2f}")
