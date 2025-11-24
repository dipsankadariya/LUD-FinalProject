# 🏠 SafeQuake 

**AI-Powered Structural Crack Detection for Earthquake Safety**

---

## 📖 About The Project

This project showcases my learnings from the Learning Utsav 2025 Challenge, where I applied the concepts and skills I gained throughout the program.
The idea for SafeQuake came randomly when I saw an article about earthquakes in our country, Nepal is highly prone to seismic activity. The Gorkha earthquake of 2015 claimed nearly 9,000 lives and destroyed over 600,000 structures.
Even today, many homes still carry hidden structural damage, and people continue living in them , which poses serious risk. Cracks and weakened surfaces can be life threatening if another earthquake strikes.

**The Problem:** Hiring a structural engineer for proper damage assessment is expensive and inaccessible for many families. Self-assessment without technical expertise is unreliable and potentially dangerous.

**The Solution:** Use the power of deep learning, so anyone with a smartphone or computer can quickly, reliably, and affordably analyze structural cracks in their home.

---

## 🎬 Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/2jRwjqF8gfw?si=p9cjpKmHQTv62wHK)

*Click above to watch the full demonstration*

### Screenshots

| Upload Interface | Analysis Results |
|:----------------:|:----------------:|
| <img src="https://github.com/user-attachments/assets/899e8595-fa0e-4f1d-85c8-fbf5c27364f0" width="300" /> | <img src="https://github.com/user-attachments/assets/370c1100-ff40-48be-a0ca-0fbbc1febeaf" width="300" /> |

| GradCAM Visualization | Severity Assessment |
|:---------------------:|:-------------------:|
| <img src="https://github.com/user-attachments/assets/cb6fe681-1460-478a-b2b0-6eaf26a95578" width="300" /> | <img src="https://github.com/user-attachments/assets/0f62c453-6761-41d8-8c5b-5f2babaec97c" width="300" /> |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SafeQuake System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Frontend   │───▶│  Flask API   │───▶│  TensorFlow/     │  │
│  │  (HTML/CSS/  │    │   Server     │    │  Keras Model     │  │
│  │     JS)      │◀───│              │◀───│  (EfficientNet)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                        │
│                    │  GradCAM Engine  │                        │
│                    │                  │                        │
│                    └──────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture & Training

### Base Model: EfficientNetB0

We use **EfficientNetB0** with transfer learning — a model that provides an excellent balance between accuracy and computational efficiency.

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────┐
│   Data Augmentation     │  ← Random flip, rotation, zoom, contrast
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ EfficientNet Preprocess │  ← Normalize pixels to [-1, 1]
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    EfficientNetB0       │  ← Pre-trained on ImageNet
│  (Convolutional Base)   │     (frozen weights, training=False)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ GlobalAveragePooling2D  │  ← Spatial dimension reduction
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   BatchNormalization    │  ← Normalize activations
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     Dropout (0.5)       │  ← Regularization
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Dense (256, ReLU)      │  ← Feature learning
│  + L2 Regularization    │     (kernel_regularizer=0.001)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   BatchNormalization    │  ← Normalize activations
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     Dropout (0.3)       │  ← Regularization
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Dense (1, Sigmoid)     │  ← Binary classification output
└───────────┬─────────────┘
            │
            ▼
      Crack / No Crack
        (0.0 - 1.0)
```

### Model Building Code

```python
def build_model(img_size=224):
    base_model = create_efficientnet_basemodel(img_size, trainable=False)
    inputs = Input(shape=(img_size, img_size, 3))
    
    x = augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return Model(inputs, outputs, name="final_model")
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 (RGB) |
| Preprocessing | EfficientNet (normalize to [-1, 1]) |
| Base Model | EfficientNetB0 (ImageNet weights, frozen) |
| Hidden Layer | Dense(256) with ReLU |
| Regularization | L2 (λ=0.001), Dropout (0.5, 0.3) |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 32 |

---

## 🔍 GradCAM

Using Gradient-weighted Class Activation Mapping (GradCAM), we visualize exactly where the model detects cracks.

### How It Works

1. **Forward Pass:** Image passes through the model to get predictions
2. **Gradient Computation:** We compute gradients of the prediction with respect to the final convolutional layer (`top_conv`)
3. **Weight Calculation:** Global average pooling of gradients gives importance weights for each feature map
4. **Heatmap Generation:** Weighted combination of feature maps, followed by ReLU activation
5. **Overlay:** Heatmap is resized and overlaid on the original image

```python
# Simplified GradCAM flow
with tf.GradientTape() as tape:
    conv_outputs, predictions = gradcam_model(img_array)
    tape.watch(conv_outputs)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)  # ReLU
```

### GradCAM Output Example

<img src="https://github.com/user-attachments/assets/cb6fe681-1460-478a-b2b0-6eaf26a95578" width="400" alt="GradCAM Example"/>

*Red/yellow regions indicate areas the model identifies as cracks*

---

## ⚠️ Severity Assessment System

Beyond binary detection, SafeQuake provides a **5-level severity assessment** based on:

- **Prediction Confidence** (50% weight)
- **Heatmap Mean Intensity** (25% weight)
- **High-Intensity Region Ratio** (25% weight)

```
Severity Score = (prediction × 50) + (heatmap_mean × 25) + (high_intensity_ratio × 25)
```

### Severity Levels

| Level | Score Range | Color | Action Required |
|-------|-------------|-------|-----------------|
| 🟢 **Safe** | < 50 (no crack) | `#22c55e` | Routine annual inspection |
| 🟡 **Minor** | < 35 | `#eab308` | Cosmetic repair within 6 months |
| 🟠 **Moderate** | 35 - 55 | `#f97316` | Professional assessment within 1 month |
| 🔴 **Severe** | 55 - 75 | `#ef4444` | Urgent evaluation within 1 week |
| ⛔ **Critical** | > 75 | `#dc2626` | **IMMEDIATE evacuation required** |

---

## 📂 Dataset

**Concrete Crack Images for Classification**

| Property | Value |
|----------|-------|
| Total Images | 40,000 |
| Categories | 2 (Crack / No Crack) |
| Image Format | RGB |
| Resolution | 224 × 224 pixels |
| Split | 80% Train / 20% Validation |

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/yatata1/crack-dataset)

---

## 📊 Model Performance

### Metrics Summary

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.91% |
| **Precision** | 0.9987 |
| **Recall** | 0.9995 |
| **F1-Score** | 0.9991 |

### Confusion Matrix

|                    | Predicted No Crack | Predicted Crack |
|--------------------|:------------------:|:---------------:|
| **Actual No Crack** | 4,075 ✓ | 5 |
| **Actual Crack** | 2 | 3,918 ✓ |

### Prediction Confidence Distribution

- **No Crack Predictions:** Mean confidence 99.77%
- **Crack Predictions:** Mean confidence 99.87%

<img src="https://github.com/user-attachments/assets/b0e5d9f8-53f6-4f75-91da-cfe3917f75ee" width="100%" alt="Prediction Distribution"/>

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.x, Flask |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Model** | EfficientNetB0 (Transfer Learning) |
| **Image Processing** | OpenCV, NumPy |
| **Explainability** | GradCAM |
| **Frontend** | HTML5, CSS3, JavaScript |
| **API** | RESTful (Flask) |

---

## 📁 Project Structure

```
SafeQuake/
├── backend/
│   ├── app.py              # Flask server & API endpoints
│   ├── models/
│   │   └── best_model.keras # Trained EfficientNetB0 model
│   └── uploads/            # Temporary upload directory
├── frontend/
│   ├── index.html          # Main web interface
│   ├── styles.css          # Styling
│   └── script.js           # Frontend logic
├── assets/                 # Images for README
├── notebooks/              # Training notebooks
├── requirements.txt        # Python dependencies
└── README.md
```

### Requirements

```txt
flask>=2.0.0
flask-cors>=3.0.0
tensorflow>=2.10.0
opencv-python>=4.5.0
numpy>=1.21.0
```

---

## 🔌 API Reference

### POST `/predict`

Analyze an image for structural cracks.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Image file (JPG, PNG)

**Response:**
```json
{
  "prediction": "Cracks Detected",
  "confidence": "98.75%",
  "crack_probability": "98.75%",
  "safe_probability": "1.25%",
  "severity": {
    "level": "Moderate",
    "score": 45.2,
    "color": "#f97316",
    "description": "Moderate structural cracks detected.",
    "recommendation": "Structural concerns present. Professional inspection advised.",
    "action_required": "Schedule professional structural assessment within 1 month"
  },
  "gradcam_image": "base64_encoded_image_string..."
}
```

---

## ⚙️ Current Status

- [x] Model training completed (99.91% accuracy)
- [x] Flask backend with REST API
- [x] GradCAM visualization integration
- [x] Severity assessment system
- [x] Web interface for image upload
- [x] Real-time prediction display

---

## 📜 License

This project is licensed under the **MIT License** 
