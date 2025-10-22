from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
models = {
    "Baseline": load_model("models/best_baseline.h5"),
    "VGG16_Transfer": load_model("models/vgg16_transfer.h5")
}

# CIFAR-10 labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get("model")
    model = models.get(model_name)

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    # Save uploaded image to static/uploads folder
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Process image for prediction
    img = Image.open(file_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_class = class_names[np.argmax(predictions)]

    # Render result.html with prediction and image
    return render_template(
        "result.html",
        model_name=model_name,
        prediction=pred_class,
        image_path=file_path
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
