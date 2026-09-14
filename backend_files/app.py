
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

MODEL_PATH = "deployment_files/pneumonia_prediction_model_v1_0.keras"  # model file is in the project directory

# Initialize Flask app
pneumonia_detection_api = Flask("pneumonia_detection_api")

# Load trained model
print("Loading pneumonia detection model...")

model = load_model(MODEL_PATH)

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
        # Open uploaded image
        image = Image.open(file).convert("RGB")

        print("Image received:", image.size)

        # Resize image to model input size
        image = image.resize((128, 128))

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch dimension
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
