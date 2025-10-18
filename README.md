# ♻️ EcoWaste AI — Smart Waste Classification & Carbon Footprint Estimator

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Tech: TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![UI: Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)

**EcoWaste AI** is an end-to-end proof-of-concept that uses computer vision and tabular ML to help users sort household waste and estimate the carbon savings from recycling or composting. Upload a photo of an item, get a classification (Organic `O` or Recyclable `R`), an estimated CO₂ saved (kg), and an eco tip.

---

## 🔎 Project Summary

- **Goal:** Automate waste sorting and estimate CO₂ savings to encourage responsible disposal and reduce landfill emissions.  
- **Inputs:** Image of waste item + estimated weight (kg).  
- **Outputs:** Predicted class (`O` or `R`), prediction confidence, estimated CO₂ saved (kg), short recycling/composting tip.  
- **Use case:** Local Streamlit app (runs on CPU), demonstrator for portfolio / sustainability hackathons.

---

![Uploading image.png…]()

---

## 🗂️ Dataset

**Source:** Kaggle — *Waste Classification Data*  
**URL:** https://www.kaggle.com/datasets/techsash/waste-classification-data

- ~25,077 images total  
  - Train: 22,564  
  - Test: 2,513  
- Classes used in this project:  
  - `O` — Organic (food / compostable)  
  - `R` — Recyclable (plastic, metal, paper, glass, etc.)

---

## 🛠️ Methodology (high level)

### 1. Data preprocessing
- Load images using `tf.keras.utils.image_dataset_from_directory`.
- Resize images to **160×160** (compatible with MobileNetV2 small alpha).
- Use `tf.keras.applications.mobilenet_v2.preprocess_input` for input normalization.

### 2. Image classification (Transfer learning)
- **Backbone:** MobileNetV2 (ImageNet weights, `alpha=0.35`) — frozen for feature extraction.
- Head: `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(2, softmax)`.
- Trained for a few epochs on the Kaggle dataset.
- Example performance: **~89% test accuracy**.

### 3. CO₂ regression (tabular)
- **Model:** `RandomForestRegressor`.
- **Features:** one-hot material columns (e.g., `material_O`, `material_R`) + `weight_kg`.
- If real labeled CO₂ data unavailable, synthetic CO₂-per-kg values were used as placeholders (replace with EPA/LCA values later).
- Example metrics: **R² ≈ 0.96**, **MSE ≈ 0.02** (on synthetic test split).

### 4. User clustering (optional)
- **Model:** `KMeans` on aggregated user upload history (features: `total_weight`, `avg_co2_saved`, `frac_O`, `frac_R`).
- Produces clusters like: *Recycle-Heavy*, *Organic Recycler*, *Mixed Recycler* — used to provide personalized tips.

### 5. App
- **Framework:** Streamlit (single `app.py` file).
- Loads models from `models/` or repo root, accepts image uploads and weight input, shows predictions, CO₂ estimate and an impact message.

---
