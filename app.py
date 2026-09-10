from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = load_model("model_makanan.h5")

# Daftar kelas makanan
classes = [
    "Bakso",
    "Sate",
    "Soto",
    "Nasi Goreng",
    "Mie Goreng",
    "Gado-gado",
    "Ayam Goreng",
    "Nasi Padang"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "NourishVision ML Service is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Image tidak ditemukan"
        }), 400

    file = request.files["image"]

    # Membaca dan memproses gambar
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array, verbose=0)[0]

    index = np.argmax(prediction)
    confidence = float(prediction[index])

    return jsonify({
        "food": classes[index],
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)