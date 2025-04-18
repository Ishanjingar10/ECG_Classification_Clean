from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import uuid
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown

# Set up environment for using CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__, template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Google Drive model
google_drive_model_url = "https://drive.google.com/uc?id=1DETKYVBjgwzSXwHZvuzpgh0eiHtllohT"
model_path = "ecg_classification_model_finetuned.h5"

def download_model():
    try:
        logging.info("Downloading model from Google Drive...")
        gdown.download(google_drive_model_url, model_path, quiet=False)
        logging.info("✅ Model downloaded successfully!")
    except Exception as e:
        logging.error(f"❌ Error downloading model: {e}")
        raise e

# Load the model
model = None
try:
    download_model()
    model = load_model(model_path, compile=False)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")

# Classes and descriptions
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

# 💡 Prediction helper function
def predict_ecg_from_path(image_path):
    try:
        if model is None:
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
            return {"error": "Prediction output mismatch with class labels."}
    except Exception as e:
        logging.exception("Prediction error")
        return {"error": f"Prediction error: {e}"}

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 🔥 Main /predict route (used by your HTML form)
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

        result = predict_ecg_from_path(file_path)

        if os.path.exists(file_path):
            os.remove(file_path)

        if "error" in result:
            logging.error(f"Prediction failed: {result['error']}")
            return jsonify(result), 500

        logging.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logging.exception("❌ Unexpected error in /predict")
        return jsonify({"error": f"Unexpected error: {e}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
