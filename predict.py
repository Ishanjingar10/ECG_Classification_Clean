import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tkinter.messagebox as msgbox
import tensorflow.keras.backend as K

# Define focal loss function (for compatibility)
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-8, 1.0)
        return -K.sum(alpha * y_true * K.log(y_pred) * (1 - y_pred) ** gamma, axis=-1)
    return loss

# Load trained model (Handle custom loss function safely)
try:
    model = load_model("ecg_classification_model_finetuned.h5", custom_objects={'loss': focal_loss()})
except Exception as e:
    model = load_model("ecg_classification_model_finetuned.h5", compile=False)
    print(f"Warning: Unable to load custom loss function. Error: {e}")

# Get input shape of the model
target_size = model.input_shape[1:3]  # Extract expected size (e.g., (128, 128))
n_channels = model.input_shape[-1]  # Extract number of channels

# Class Labels and Descriptions
class_labels = [
    "Fusion Beats (F)",
    "Miscellaneous Beats (M)",
    "Normal Beats (N)",
    "Unknown Beats / Noise (Q)",
    "Supraventricular Beats (S)",
    "Ventricular Beats (V)"
]

class_descriptions = {
    "Fusion Beats (F)": "A combination of normal and abnormal beats, often seen in conditions like Ventricular Fusion Beats.",
    "Miscellaneous Beats (M)": "Irregular beats that do not fit standard categories, possibly indicating rare conduction disorders.",
    "Normal Beats (N)": "A healthy heart rhythm with no detected abnormalities. Heart rate typically ranges from **60-100 bpm**.",
    "Unknown Beats / Noise (Q)": "Unrecognized signals due to artifacts or improper ECG recording. May include pacemaker beats.",
    "Supraventricular Beats (S)": "Abnormal beats from atria or AV junction, seen in **AFib, Atrial Flutter, and SVT (150-250 bpm)**.",
    "Ventricular Beats (V)": "Serious arrhythmias like **PVCs, Ventricular Tachycardia (VT 100-250 bpm), and VFib (cardiac arrest risk).**"
}

# Initialize GUI
root = tk.Tk()
root.title("ECG Classification System")
root.geometry("450x700")

# Label to show the selected image
img_label = Label(root, text="Select an ECG Image", font=("Arial", 12))
img_label.pack(pady=10)

# Label to display prediction result
result_label = Label(root, text="", font=("Arial", 12), fg="blue", wraplength=400, justify="left")
result_label.pack(pady=10)

# Function to process and predict ECG
def predict_ecg(image_path):
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=target_size)  # Load as RGB or grayscale
        img_array = img_to_array(img) / 255.0  # Normalize

        # Ensure correct number of channels (convert grayscale to RGB if needed)
        if img_array.shape[-1] == 1 and n_channels == 3:
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 3 and n_channels == 1:
            img_array = np.mean(img_array, axis=-1, keepdims=True)  # Convert RGB to grayscale

        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict with the model
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_idx]
        confidence = np.max(prediction) * 100
        description = class_descriptions.get(predicted_class, "No description available.")

        # Debugging confidence values
        print(f"Raw Model Output: {prediction}")
        print(f"Predicted: {predicted_class}, Confidence: {confidence:.2f}%")

        # Show Warning for Low Confidence Predictions
        if confidence < 50:
            msgbox.showwarning("Low Confidence", f"The model is unsure of the prediction.\nConfidence: {confidence:.2f}%")

        # Update the result label in the GUI
        result_text = f"🔹 **Predicted Class:** {predicted_class}\n🔹 **Confidence:** {confidence:.2f}%\n\n📝 **Description:**\n{description}"
        result_label.config(text=result_text)
    
    except Exception as e:
        msgbox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

# Function to open file dialog and update GUI
def select_image():
    file_path = filedialog.askopenfilename(title="Select an ECG Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img, text="")  # Hide text when image is loaded
            img_label.image = img  # Keep reference
            predict_ecg(file_path)
        except Exception as e:
            msgbox.showerror("Image Error", f"Failed to load image:\n{e}")
    else:
        msgbox.showwarning("Warning", "No file selected.")

# Button to select an image
select_button = Button(root, text="Select ECG Image", command=select_image, font=("Arial", 12), bg="lightblue")
select_button.pack(pady=10)

# Run GUI
root.mainloop()
