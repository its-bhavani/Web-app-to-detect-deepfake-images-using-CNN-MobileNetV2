import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import os

# Define dataset paths
train_dir = "train_balanced/"
test_dir = "test_balanced/"

# Image Data Augmentation for Better Generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
)

# ✅ Option 1: Use Transfer Learning (MobileNetV2) [Recommended]
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Use Adam Optimizer for Better Training
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model for 50 Epochs (More Training for Better Accuracy)
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Save Model in the Recommended `.keras` Format
model.save("deepfake_model_balanced.keras")
print("✅ Model training complete! Saved as 'deepfake_model_balanced.keras'.")
