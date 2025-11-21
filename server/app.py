from flask import Flask, jsonify, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import efficientnet
import numpy as np
import os
from flask_cors import CORS
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)
CORS(app)

# first, lets load the model
MODEL_PATH = "./models/best_model.keras"
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# then lets create temporary directory named 'uploads' where we will save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------- #
#       GRAD-CAM FUNCTION         #
# ------------------------------- #
def generate_gradcam(img_array, last_conv_layer_name=None):
    """
    Returns Grad-CAM heatmap (3-channel RGB).
    """
    # If user didn’t specify, automatically detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layer (batch, h, w, channels)
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]  # crack class

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize 0–1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def overlay_gradcam(original, heatmap, alpha=0.4):
    """
    Overlay heatmap on original RGB image.
    """
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # convert to RGB
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlay


def encode_image_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')


# ------------------------------- #
#        SEVERITY SCORE           #
# ------------------------------- #
def get_severity(prob):
    if prob < 0.35:
        return "Mild"
    elif prob < 0.65:
        return "Moderate"
    else:
        return "Severe"


# ------------------------------- #
#         PREDICT ROUTE           #
# ------------------------------- #
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # lets handle the file upload first 
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        # save the file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # load image in RGB mode
        img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
        img_array = image.img_to_array(img)

        # Grad-CAM needs original 0–255 RGB
        original_img = img_array.astype(np.uint8)

        # expand dims and preprocess for EfficientNet
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efficientnet.preprocess_input(img_array)

        # make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        result = "Cracks Detected" if prediction > 0.5 else "No Cracks Detected"
        confidence = float(prediction if prediction > 0.5 else (1 - prediction))

        # ------------------------------- #
        #           GRAD-CAM              #
        # ------------------------------- #
        heatmap = generate_gradcam(img_array)
        overlay = overlay_gradcam(original_img, heatmap)

        gradcam_b64 = encode_image_base64(overlay)

        # ------------------------------- #
        #         SEVERITY SCORE          #
        # ------------------------------- #
        severity = get_severity(float(prediction))

        # clean up uploaded file 
        os.remove(file_path)

        # return the json
        return jsonify({
            "prediction": result,
            "confidence": f"{confidence * 100:.2f}%",
            "crack_probability": f"{prediction * 100:.2f}%",
            "safe_probability": f"{(1 - prediction) * 100:.2f}%",
            "severity_level": severity,
            "gradcam_image": gradcam_b64
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# serve the HTML homepage
@app.route("/")
def home():
    return send_from_directory('../frontend', 'index.html')

# serve static files (CSS, JS, images, etc.)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory('../frontend', filename)

if __name__ == "__main__":
    app.run(debug=True)
