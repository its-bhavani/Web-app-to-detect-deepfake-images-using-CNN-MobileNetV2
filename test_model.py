import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("deepfake_model_balanced.keras")

# Function to preprocess an image before feeding it into the model
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"🚨 Error: Image file '{image_path}' not found!")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"🚨 Error: Unable to read image file '{image_path}'.")
        return None

    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values (0 to 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Test the model with a sample image
sample_image = "test_balanced/Real/00278.jpg"  # Change this path to an actual image in your dataset
image = preprocess_image(sample_image)

if image is None:
    print("❌ Test failed: Image not found or unreadable.")
else:
    prediction = model.predict(image)[0][0]
    print(f"Prediction Score: {prediction}")

    # Adjust threshold to make predictions more accurate
    threshold = 0.6
    result = "🚨 Deepfake Detected!" if prediction > threshold else "✅ Real Face Detected!"
    print(f"Result: {result}")
