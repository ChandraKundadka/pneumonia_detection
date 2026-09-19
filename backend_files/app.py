
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os
import pydicom


# Initialize Flask app
pneumonia_detection_api = Flask("pneumonia_detection_api")

#Prepare model path
MODEL_REPO_ID = os.getenv(
    "MODEL_REPO_ID",
    "Chandrashekhara/pneumonia_prediction_model",
)
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "pneumonia_prediction_model_MobileNetV2_ffnn.keras",
)

# Load trained model
print("Loading pneumonia detection model...")

# For security, pull the token from environment variables instead of hardcoding
hf_token = os.getenv("HF_TOKEN")

# Download the file to the local cache directory

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
    repo_type="model",  # Use "dataset" if it's stored in a dataset repo, or "model" (default)
    token=hf_token
)

# Load the local model file
model = load_model(model_path)

print("Model loaded successfully!")


# Home route
@pneumonia_detection_api.get("/")
def home():
    return "Welcome to the Pneumonia Detection API!"


# Health check
@pneumonia_detection_api.get("/health")
def health():
    return jsonify({"status": "ok"})


# Pneumonia prediction endpoint
@pneumonia_detection_api.post("/v1/predict")
def predict_pneumonia():

    print("Prediction request received")

    # Check whether an image was uploaded
    if "file" not in request.files:
        return jsonify({
            "error": "No image file provided"
        }), 400

    file = request.files["file"]

    # Check filename
    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        filename = file.filename.lower()

        if filename.endswith(".dcm"):
            # Read DICOM file directly from the file stream
            dicom_data = pydicom.dcmread(file.stream)
            pixel_array = dicom_data.pixel_array.astype(float)

        # Normalize DICOM pixel values to [0, 1] safely
            denom = pixel_array.max() - pixel_array.min()
            if denom > 0:
                pixel_array = (pixel_array - pixel_array.min()) / denom
            else:
                pixel_array = np.zeros_like(pixel_array)

            # Convert to 3-channel RGB PIL Image (scaled to 0-255 for standard PIL resizing)
            uint8_image = (pixel_array * 255.0).astype(np.uint8)
            image = Image.fromarray(uint8_image).convert("RGB")

        else:
            # Standard formats: PNG, JPG, JPEG, etc.
            image = Image.open(file.stream).convert("RGB")

        print("Image received:", image.size)

# Preprocessing
        # Resize image to model input size
        image = image.resize((128, 128))

        # Normalize to [0, 1] and add batch dimension (1, 128, 128, 3)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        print("Image preprocessed")

# Model prediction
        prediction = model.predict(image_array)

        print("Raw prediction:", prediction)

        # -----------------------------------------
        # IMPORTANT:
        # This assumes sigmoid output:
        # 0 = NORMAL
        # 1 = PNEUMONIA
        # -----------------------------------------

        probability = float(prediction[0][0])

        if probability >= 0.5:
            predicted_class = "Pneumonia"
            confidence = probability
        else:
            predicted_class = "Normal"
            confidence = 1 - probability

        print("Prediction:", predicted_class)
        print("Confidence:", confidence)

        # Return JSON response
        return jsonify({
            "prediction": predicted_class,
            "pneumonia_probability": probability,
            "confidence": confidence
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# Run Flask app
if __name__ == "__main__":
    pneumonia_detection_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
