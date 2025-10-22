from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import traceback

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
        "VGG16_Transfer": load_model
