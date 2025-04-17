import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K

# Define paths
dataset_path = "ECG_Image_data/"
img_size = (128, 128)
batch_size = 32
epochs = 20  # Initial Training
fine_tune_epochs = 15  # Fine-Tuning

# Data Preprocessing (Stronger Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False  # ECGs should not be flipped
)

train_data = datagen.flow_from_directory(
    dataset_path + "train/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path + "train/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Compute Class Weights to Handle Imbalance
class_labels = list(train_data.class_indices.keys())
y_train = np.concatenate([train_data.classes])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("Computed Class Weights:", class_weights_dict)

# Load Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the model initially

# Define Custom Model
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-8, 1.0)
        return -K.sum(alpha * y_true * K.log(y_pred) * (1 - y_pred) ** gamma, axis=-1)
    return loss

# Build Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Normalize activations
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)  # Stronger dropout
x = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=focal_loss(),
              metrics=['accuracy'])

# Callbacks to Reduce Overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train Model (Initial Training)
history = model.fit(train_data, validation_data=val_data, epochs=epochs, 
                    callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict)

# Fine-Tuning: Unfreeze Deeper Layers
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
    layer.trainable = True

# Recompile with Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=focal_loss(),
              metrics=['accuracy'])

# Train Model (Fine-Tuning)
history_finetune = model.fit(train_data, validation_data=val_data, epochs=fine_tune_epochs,
                             callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict)

# Save the Model
model.save("ecg_classification_model_finetuned.h5")
print("Model saved as ecg_classification_model_finetuned.h5")
