from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import traceback
import uuid

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# Debug: Print contents of models folder
print("Model folder contents:", os.listdir("models"))
for fname in os.listdir("models"):
    print(fname, "| size:", os.path.getsize(os.path.join("models", fname)))

app = Flask(__name__)

# Try loading models and log errors
try:
    models = {
        "Baseline": load_model("models/best_baseline.h5"),
        "VGG16_Transfer": load_model("models/vgg16_transfer.h5")
    }
    print("Models loaded successfully.")
except Exception as e:
    print("MODEL LOAD FAILURE:", traceback.format_exc())
    models = {}

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get("model")
        print("Model requested:", model_name)
        model = models.get(model_name)
        if model is None:
            raise ValueError("Model not loaded or not found!")

        if "file" not in request.files:
            raise ValueError("No file uploaded.")

        file = request.files["file"]
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        # Save with a unique filename to prevent overwrite
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        print("File saved to:", file_path)

        # Preprocess the image
        img = Image.open(file_path).convert("RGB")
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("Image shape:", img_array.shape)
        predictions = model.predict(img_array)
        pred_class = class_names[np.argmax(predictions)]
        print("Prediction:", pred_class)

        return render_template(
            "result.html",
            model_name=model_name,
            prediction=pred_class,
            image_path=file_path
        )
    except Exception as e:
        error_message = traceback.format_exc()
        print("ERROR in /predict:", error_message)
        return f"<pre>{error_message}</pre>", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
