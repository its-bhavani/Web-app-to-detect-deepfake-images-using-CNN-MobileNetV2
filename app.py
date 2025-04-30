from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained deepfake detection model
model = tf.keras.models.load_model("deepfake_model_balanced.keras")

# Define the folder for uploaded images
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess uploaded images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Preprocess and predict
            image = preprocess_image(file_path)
            if image is None:
                return render_template("index.html", result="Error: Invalid image file!", image=None)

            prediction = model.predict(image)[0][0]
            threshold = 0.6 # Adjust threshold for better accuracy
            result = "🚨 Deepfake Detected!" if prediction > threshold else "✅ Real Face Detected !"

            return render_template("index.html", result=result, image=file_path)

    return render_template("index.html", result=None, image=None)

if __name__ == "__main__":
    app.run(debug=True)
