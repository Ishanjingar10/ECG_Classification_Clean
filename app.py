# app.py (Flask backend)

import os
import logging
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown  # <- Make sure this is installed (pip install gdown)

# Set up environment variables for Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model path and Google Drive file ID from environment variables
model_path = "ecg_classification_model_finetuned.h5"
drive_file_id = os.getenv("MODEL_DRIVE_FILE_ID", "1DETKYVBjgwzSXwHZvuzpgh0eiHtllohT")  # Use environment variable for Drive file ID

# Download model from Google Drive if not present
if not os.path.exists(model_path):
    logging.info("🔽 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)
else:
    logging.info("📁 Model file found locally.")

# Load the model
model = None
try:
    model = load_model(model_path, compile=False)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")

# Class labels and descriptions
class_info = {
    "Fusion Beats (F)": "A combination of normal and abnormal beats, often seen in conditions like Ventricular Fusion Beats.",
    "Miscellaneous Beats (M)": "Irregular beats that do not fit standard categories.",
    "Normal Beats (N)": "A healthy heart rhythm with no detected abnormalities.",
    "Unknown Beats / Noise (Q)": "Unrecognized signal patterns, possibly due to noise or improper lead placement.",
    "Supraventricular Beats (S)": "Abnormal beats originating from the atria or AV junction.",
    "Ventricular Beats (V)": "Abnormal beats originating from the ventricles, often indicating serious arrhythmias."
}
class_labels = list(class_info.keys())

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction logic
def predict_ecg(image_path):
    try:
        if model is None:
            logging.error("Model is not loaded.")
            return {"error": "Model not loaded."}

        img = load_img(image_path, target_size=(128, 128), color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        logging.info(f"Raw prediction: {prediction}")

        if isinstance(prediction, np.ndarray) and prediction.shape[1] == len(class_labels):
            predicted_index = int(np.argmax(prediction))
            predicted_class = class_labels[predicted_index]
            confidence = float(np.max(prediction) * 100)
            description = class_info.get(predicted_class, "No description available.")
            return {
                "class": predicted_class,
                "confidence": confidence,
                "description": description
            }
        else:
            logging.error("Prediction output shape mismatch.")
            return {"error": "Prediction output mismatch with class labels."}

    except Exception as e:
        logging.exception("Prediction error")
        return {"error": f"Prediction error: {e}"}

# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Upload and prediction route
@app.route("/predict", methods=["POST"])
def upload_and_predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG."}), 400

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        file.save(file_path)
        logging.info(f"Image saved to {file_path}")

        result = predict_ecg(file_path)

        if os.path.exists(file_path):
            os.remove(file_path)

        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)

    except Exception as e:
        logging.exception("❌ Unexpected error in /predict")
        return jsonify({"error": f"Unexpected error: {e}"}), 500

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not provided by Railway
    app.run(debug=False, host="0.0.0.0", port=port)
