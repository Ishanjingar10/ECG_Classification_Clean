from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import uuid
import os
import numpy as np
import gc
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB max upload size
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model path
MODEL_URL = "https://drive.google.com/uc?id=1DETKYVBjgwzSXwHZvuzpgh0eiHtllohT"
MODEL_PATH = "ecg_classification_model_finetuned.h5"
model = None

# ECG Class descriptions
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

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        logging.info("✅ Model downloaded.")

def get_model():
    global model
    if model is None:
        download_model()
        model = load_model(MODEL_PATH, compile=False)
        logging.info("✅ Model loaded into memory.")
    return model

def predict_ecg_from_path(image_path):
    try:
        model = get_model()

        img = load_img(image_path, target_size=(128, 128), color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        logging.info(f"Prediction raw output: {prediction}")

        if prediction.shape[1] != len(class_labels):
            return {"error": "Prediction mismatch with class labels."}

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(prediction) * 100)

        result = {
            "class": predicted_class,
            "confidence": round(confidence, 2),
            "description": class_info[predicted_class]
        }

        # Cleanup
        del img, img_array, prediction
        gc.collect()

        return result

    except Exception as e:
        logging.exception("Error during prediction")
        return {"error": str(e)}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Use PNG, JPG, or JPEG."}), 400

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = predict_ecg_from_path(filepath)

        # Remove uploaded image
        os.remove(filepath)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        logging.exception("Error handling /predict route")
        return jsonify({"error": f"Unexpected error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
