from flask import Flask, jsonify, request, send_from_directory
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import efficientnet
import tensorflow as tf
import numpy as np
import os
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# first, lets load the model
MODEL_PATH = "./models/best_model.keras"
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# then lets create temporary directory named 'uploads' where we will save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============== Build a fresh GradCAM-compatible model ==============
print("Building GradCAM model...")

# Get EfficientNet base
efficientnet_base = model.get_layer('efficientnetb0')

# Create a new combined model that outputs both conv features and final prediction
# Input
inp = model.input

# Pass through data augmentation
data_aug = model.get_layer('data_augmentation')
x = data_aug(inp)

# Get EfficientNet outputs - we need both the conv features and the final output
# Create a model from efficientnet that gives us the top_conv output
top_conv_output = efficientnet_base.get_layer('top_conv').output
efficientnet_with_conv = Model(
    inputs=efficientnet_base.input,
    outputs=[top_conv_output, efficientnet_base.output]
)

# Pass augmented input through efficientnet
conv_features, eff_output = efficientnet_with_conv(x)

# Continue through rest of the model
x = model.get_layer('global_average_pooling2d_2')(eff_output)
x = model.get_layer('batch_normalization_4')(x)

# Find and apply remaining layers (dropout and dense)
for layer in model.layers:
    if 'dropout' in layer.name.lower():
        x = layer(x)
    elif 'dense' in layer.name.lower():
        x = layer(x)

# Create the GradCAM model
gradcam_model = Model(inputs=inp, outputs=[conv_features, x])
print("GradCAM model built successfully!")


def get_gradcam_heatmap(img_array):
    """Generate GradCAM heatmap using the custom gradcam_model."""
    try:
        with tf.GradientTape() as tape:
            # Get conv outputs and predictions
            conv_outputs, predictions = gradcam_model(img_array, training=False)
            tape.watch(conv_outputs)
            
            # For binary classification with sigmoid
            loss = predictions[:, 0]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            print("Warning: Gradients are None")
            return np.ones((7, 7)) * 0.5
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get conv outputs without batch dim
        conv_outputs = conv_outputs[0]
        
        # Weight channels by gradient importance and sum
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # ReLU - only keep positive contributions
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize to 0-1
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.ones_like(heatmap) * 0.5
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"GradCAM error: {e}")
        import traceback
        traceback.print_exc()
        return np.ones((7, 7)) * 0.5


def apply_gradcam_to_image(original_img_path, heatmap, alpha=0.5):
    """Overlay GradCAM heatmap on the original image."""
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blend
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def image_to_base64(img_array):
    """Convert numpy image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')


def assess_severity(prediction, heatmap):
    """Assess crack severity based on prediction and heatmap intensity."""
    heatmap_mean = np.mean(heatmap)
    high_intensity_ratio = np.sum(heatmap > 0.5) / heatmap.size
    
    severity_score = (
        prediction * 50 +
        heatmap_mean * 25 +
        high_intensity_ratio * 25
    ) * 100
    
    if prediction < 0.5:
        return {
            "level": "Safe",
            "score": round(severity_score, 1),
            "color": "#22c55e",
            "description": "No significant cracks detected.",
            "recommendation": "Building appears structurally sound. Continue regular maintenance.",
            "action_required": "None - routine inspection recommended annually"
        }
    elif severity_score < 35:
        return {
            "level": "Minor",
            "score": round(severity_score, 1),
            "color": "#eab308",
            "description": "Minor surface cracks detected.",
            "recommendation": "Superficial damage observed. Monitor for progression.",
            "action_required": "Cosmetic repair recommended within 6 months"
        }
    elif severity_score < 55:
        return {
            "level": "Moderate",
            "score": round(severity_score, 1),
            "color": "#f97316",
            "description": "Moderate structural cracks detected.",
            "recommendation": "Structural concerns present. Professional inspection advised.",
            "action_required": "Schedule professional structural assessment within 1 month"
        }
    elif severity_score < 75:
        return {
            "level": "Severe",
            "score": round(severity_score, 1),
            "color": "#ef4444",
            "description": "Severe structural damage detected.",
            "recommendation": "Significant structural compromise. Immediate action required.",
            "action_required": "Urgent structural engineering evaluation required within 1 week"
        }
    else:
        return {
            "level": "Critical",
            "score": round(severity_score, 1),
            "color": "#dc2626",
            "description": "Critical structural failure risk.",
            "recommendation": "Dangerous structural damage. Evacuation may be necessary.",
            "action_required": "IMMEDIATE evacuation and emergency structural assessment required"
        }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array_processed = efficientnet.preprocess_input(img_array.copy())

        # Make prediction
        prediction = model.predict(img_array_processed, verbose=0)[0][0]
        result = "Cracks Detected" if prediction > 0.5 else "No Cracks Detected"
        confidence = float(prediction if prediction > 0.5 else (1 - prediction))

        # Generate GradCAM
        gradcam_base64 = None
        severity = None
        try:
            print("Generating GradCAM...")
            heatmap = get_gradcam_heatmap(img_array_processed)
            print(f"Heatmap - min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, mean: {heatmap.mean():.4f}")
            
            gradcam_img = apply_gradcam_to_image(file_path, heatmap)
            gradcam_base64 = image_to_base64(gradcam_img)
            severity = assess_severity(float(prediction), heatmap)
        except Exception as e:
            print(f"GradCAM error: {e}")
            import traceback
            traceback.print_exc()
            severity = assess_severity(float(prediction), np.ones((7, 7)) * 0.5)

        os.remove(file_path)

        print(f"Result: {result}, Confidence: {confidence * 100:.2f}%")
        if severity:
            print(f"Severity: {severity['level']} (Score: {severity['score']})")

        response_data = {
            "prediction": result,
            "confidence": f"{confidence * 100:.2f}%",
            "crack_probability": f"{prediction * 100:.2f}%",
            "safe_probability": f"{(1 - prediction) * 100:.2f}%",
            "severity": severity
        }
        
        if gradcam_base64:
            response_data["gradcam_image"] = gradcam_base64

        return jsonify(response_data), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return send_from_directory('../frontend', 'index.html')


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory('../frontend', filename)


if __name__ == "__main__":
    app.run(debug=True)