import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Define focal loss function (for compatibility)
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-8, 1.0)
        return -K.sum(alpha * y_true * K.log(y_pred) * (1 - y_pred) ** gamma, axis=-1)
    return loss

# Load trained model (Ensure compatibility with custom loss)
try:
    model = load_model("ecg_classification_model_finetuned.h5", custom_objects={'loss': focal_loss()})
except:
    model = load_model("ecg_classification_model_finetuned.h5")

# Data Preprocessing for Testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "ECG_Image_data/test/",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate Model
loss, accuracy = model.evaluate(test_data)
print(f"\n🔥 Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Generate Predictions
y_pred_probs = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Display Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# Confidence Scores for Predictions
confidences = np.max(y_pred_probs, axis=1)
print("\n🔍 Sample Confidence Scores:")
for i in range(5):  # Print first 5 samples
    print(f"True: {class_labels[y_true_classes[i]]}, Predicted: {class_labels[y_pred_classes[i]]}, Confidence: {confidences[i] * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
