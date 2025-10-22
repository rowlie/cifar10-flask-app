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
    models = {}
    
    # Check if model files exist before loading
    if os.path.exists("models/best_baseline.h5"):
        models["Baseline"] = load_model("models/best_baseline.h5")
        print("Baseline model loaded successfully.")
    else:
        print("WARNING: models/best_baseline.h5 not found!")
    
    if os.path.exists("models/vgg16_transfer.h5"):
        models["VGG16_Transfer"] = load_model("models/vgg16_transfer.h5")
        print("VGG16 Transfer model loaded successfully.")
    else:
        print("WARNING: models/vgg16_transfer.h5 not found!")
    
    if models:
        print(f"Successfully loaded {len(models)} model(s): {list(models.keys())}")
    else:
        print("ERROR: No models were loaded. The app will not work properly.")
        
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
        print(f"Model requested: {model_name}")
        
        # Check if model exists
        model = models.get(model_name)
        if model is None:
            available_models = list(models.keys())
            raise ValueError(
                f"Model '{model_name}' not loaded or not found! "
                f"Available models: {available_models}"
            )

        if "file" not in request.files:
            raise ValueError("No file uploaded.")

        file = request.files["file"]
        
        # Check if file was actually selected
        if file.filename == "":
            raise ValueError("No file selected.")
        
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        # Save with a unique filename to prevent overwrite
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Preprocess the image
        img = Image.open(file_path).convert("RGB")
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print(f"Image shape: {img_array.shape}")
        
        # Make prediction
        predictions = model.predict(img_array)
        pred_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        print(f"Prediction: {pred_class} (confidence: {confidence:.2%})")

        # Pass only the filename (not the full path) to the template
        return render_template(
            "result.html",
            model_name=model_name,
            prediction=pred_class,
            image_path=unique_filename  # FIXED: Pass only filename, not full path
        )
        
    except Exception as e:
        error_message = traceback.format_exc()
        print("ERROR in /predict:", error_message)
        return f"<h1>Error Processing Request</h1><pre>{error_message}</pre>", 500

if __name__ == "__main__":
    # This section only runs when executing the script directly (for local development)
    # Render uses gunicorn to start the app, so this won't execute in production
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # FIXED: Use PORT env var, disable debug
