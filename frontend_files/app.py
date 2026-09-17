
import streamlit as st
import requests
from PIL import Image
import io
import pydicom
import numpy as np
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

#BACKEND_URL = "https://Chandrashekhara-superkart-sales-predictor-backend.hf.space"

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="@",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Pneumonia Detection App")

st.write(
    "Upload a chest X-ray image and the AI model will "
    "classify it as Normal or Pneumonia."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["dcm", "jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Display uploaded image
# --------------------------------------------------

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
# Read DICOM using pydicom

    if filename.endswith(".dcm"):
        dcm = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
        raw_arr = dcm.pixel_array.astype(np.float32)

        # Normalize pixel values to [0, 255] uint8 for display
        pixel_range = raw_arr.max() - raw_arr.min()
        if pixel_range == 0:
            norm_arr = np.zeros_like(raw_arr, dtype=np.uint8)
        else:
            norm_arr = ((raw_arr - raw_arr.min()) / pixel_range * 255).astype(np.uint8)

        display_image = norm_arr
    else:
        # Standard raster image (PNG, JPG, JPEG)
        display_image = Image.open(uploaded_file)


    st.subheader("Uploaded X-ray")

    st.image(
        display_image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )


    # --------------------------------------------------
    # Prediction button
    # --------------------------------------------------

    if st.button(
        "Detect Pneumonia",
        type="primary"
    ):

        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Send image to backend
            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                },
                timeout=60
            )


            # --------------------------------------------------
            # Successful response
            # --------------------------------------------------

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                pneumonia_probability = result[
                    "pneumonia_probability"
                ]

                st.subheader("Prediction")
                if prediction == "Pneumonia":

                    st.error(
                        f"🫁 Pneumonia detected\n\n"
                        f"Confidence: {confidence * 100:.2f}%"
                    )
                else:
                    st.success(
                        f"✅ Normal\n\n"
                        f"Confidence: {confidence * 100:.2f}%"
                    )

                # Show probability
                st.write(
                    f"**Pneumonia probability:** "
                    f"{pneumonia_probability * 100:.2f}%"
                )


                # Progress bar
                st.progress(
                    pneumonia_probability
                )

            # --------------------------------------------------
            # Backend error
            # --------------------------------------------------
            else:

                try:
                    error_message = response.json().get(
                        "error",
                        "Unknown backend error"
                    )
                except Exception:
                    error_message = response.text

                st.error(
                    f"Backend error "
                    f"({response.status_code}): "
                    f"{error_message}"
                )


        except requests.exceptions.Timeout:

            st.error(
                "The backend took too long to respond. "
                "Please try again."
            )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the pneumonia "
                "detection backend."
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )
