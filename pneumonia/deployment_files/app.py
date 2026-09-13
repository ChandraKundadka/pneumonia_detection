import io
import os
import cv2
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
IMG_SIZE = 128
ALLOWED_EXTENSIONS = ['dcm', 'jpg', 'jpeg', 'png', 'bmp', 'webp']
MODEL_PATH = "deployment_files/pneumonia_prediction_model_v1_0.keras"  # model file is in the project directory

st.set_page_config(page_title="Pneumonia Detection", layout="centered") 


@st.cache_resource
def load_pneumonia_model():
    """Load and cache the trained Keras model."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_pneumonia_model()


# ==========================================
# 2. Image Preprocessing Logic (from Notebook)
# ==========================================
def process_uploaded_image(raw_bytes, filename, img_size=IMG_SIZE, channels=3):
    """
    Decodes uploaded file bytes (supporting DICOM and standard formats),
    normalizes bit-depth, resizes, and creates model-ready tensors.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.dcm':
        # 1. Parse raw bytes as DICOM
        dcm_data = pydicom.dcmread(io.BytesIO(raw_bytes))
        arr = dcm_data.pixel_array.astype(np.float32)

        # Normalize 12/16-bit medical grayscale to 8-bit [0, 255]
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        arr = arr.astype(np.uint8)

    elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
        # Standard image decoding via PIL
        img = Image.open(io.BytesIO(raw_bytes)).convert('L')
        arr = np.array(img, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # 2. Resize to 128x128
    arr_resized = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 3. Convert grayscale to 3-channel RGB
    if channels == 3:
        arr_rgb = cv2.cvtColor(arr_resized, cv2.COLOR_GRAY2RGB)
    else:
        arr_rgb = arr_resized[:, :, np.newaxis]

    # Return displayable 8-bit image and batch-expanded tensor
    tensor = np.expand_dims(arr_rgb, axis=0)
    return arr, tensor


# ==========================================
# 3. Streamlit UI
# ==========================================
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a medical chest X-ray in **DICOM (.dcm)** or standard image format (`.png`, `.jpg`, `.jpeg`).")

if model is None:
    st.warning(f"⚠️ Model file `{MODEL_PATH}` was not found. Inference will be disabled until you place your model in this folder.")

uploaded_file = st.file_uploader(
    "Choose an X-Ray file", 
    type=ALLOWED_EXTENSIONS
)

if uploaded_file is not None:
    try:
        # Streamlit UploadedFile provides read() directly
        raw_bytes = uploaded_file.read()

        with st.spinner("Processing image and running diagnosis..."):
            original_arr, input_tensor = process_uploaded_image(
                raw_bytes, 
                uploaded_file.name, 
                img_size=IMG_SIZE, 
                channels=3
            )

            # Display uploaded image preview
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original_arr, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            with col2:
                st.subheader("Results")
                if model is not None:
                    # Model inference
                    prediction = model.predict(input_tensor)
                    prob = float(prediction[0][0])
                    
                    if prob >= 0.5:
                        st.error(f"**Diagnosis: Pneumonia / Lung Opacity**")
                    else:
                        st.success(f"**Diagnosis: Normal**")
                    
                    st.metric(label="Probability Score", value=f"{prob:.2%}")
                    st.progress(prob)
                else:
                    st.info(f"Image successfully preprocessed into tensor with shape `{input_tensor.shape}`.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
