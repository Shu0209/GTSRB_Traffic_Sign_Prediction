import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "ppm", "bmp", "webp"}
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/traffic_sign_model.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "traffic_data/labels.csv")
IMG_SIZE = (32, 32)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# Load model + label map at startup
model = None
label_map = {}


def load_resources():
    global model, label_map

    global model, label_map

    print("=" * 50)
    print("Current Directory:", os.getcwd())
    print("MODEL_PATH:", MODEL_PATH)
    print("Exists:", os.path.exists(MODEL_PATH))
    print("=" * 50)

    try:
        model = load_model(MODEL_PATH)
        print("[SUCCESS] Model loaded")
    except Exception as e:
        print("[ERROR]", e)

        
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {MODEL_PATH}. Predictions will be unavailable.")

    if os.path.exists(LABELS_PATH):
        import pandas as pd
        df = pd.read_csv(LABELS_PATH)
        label_map = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        print(f"[INFO] Loaded {len(label_map)} class labels.")
    else:
        # Fallback: numeric labels
        label_map = {i: f"Class {i}" for i in range(43)}
        print("[WARNING] labels.csv not found. Using numeric class IDs.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Could not read image file.")
    image = cv2.resize(image, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)  # shape: (1, 32, 32, 3)


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Place traffic_sign_model.keras at the configured path."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        img_tensor = preprocess_image(save_path)
        preds = model.predict(img_tensor, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        confidence = float(preds[pred_class])

        # Top-5 predictions
        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [
            {"class_id": int(i), "label": label_map.get(int(i), f"Class {i}"), "confidence": round(float(preds[i]) * 100, 2)}
            for i in top5_idx
        ]

        return jsonify({
            "success": True,
            "image_url": f"/{save_path}",
            "prediction": {
                "class_id": pred_class,
                "label": label_map.get(pred_class, f"Class {pred_class}"),
                "confidence": round(confidence * 100, 2),
            },
            "top5": top5,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
load_resources()
if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=5000)