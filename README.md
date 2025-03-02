---
title: Apnea Detector
emoji: 🏢
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
license: mit
short_description: Detect Bradypnea, Tachypnea, Apnea, Normal breathing rate.
---

- Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
- Check out the model at https://huggingface.co/spaces/AbdullahNasir/Apnea-Detector

Breathing rate (*bpm - breaths per minute*) is a crucial health indicator. Traditional methods require **wearable sensors**, which can be uncomfortable. This project explores a **non-contact, video-based approach** using deep learning and time series forecasting to estimate **bpm from chest movement** and detect anomalies in respiratory patterns. 
# **Breathing Rate Estimation and Anomaly Detection from Video Data**  
📌 *Deep Learning-Based Non-Contact Respiratory Monitoring*  

![Breathing Rate Estimation](https://raw.githubusercontent.com/Abdullah-Nasir-Chowdhury/Apnea-Detector/refs/heads/main/images/3.jpeg) 
## 🚀 **Introduction**  
 

## 🎯 **Objectives**  
✔ Estimate breathing rate from **video data**  
✔ Detect **anomalies** in respiratory patterns  
✔ Develop a **deep learning ensemble model** for bpm prediction  
✔ Use **SHAP & Grad-CAM** for explainability and visualization  

---

## 📂 **Dataset Overview**  
- **40 videos (~2 min each)**, varying in length  
- Ground truth bpm recorded every **10 seconds** via a **chest-strapped device**  
- CSV files containing:  
  - ⏳ `time` (every 10s)  
  - 💨 `bpm` (ground truth)  
- **Force data** was later introduced for correlation analysis  

---

## 🛠 **Methodology**  

### 📌 **1. Data Preprocessing**  
- **Region of Interest (ROI)**: Focused on **chest movement**  
- **Optical Flow Tracking**: Used **Farneback, Lucas-Kanade, RAFT**  
- **Peak Detection**: Extracted peaks/troughs for breathing cycles  
- **Feature Extraction**: Converted motion signals into structured data  
- **Normalization**: Applied selectively to avoid distorting bpm values  

### 📌 **2. Time Series Prediction Models**  
We experimented with:  
- **TabNet** – Interpretable tabular deep learning model  
- **Temporal Fusion Transformer (TFT)** – Long-range dependency capture  
- **Temporal Convolutional Network (TCN)** – Robust time series forecasting  
- **ETSformer** – Exponential Smoothing & Transformer-based prediction  
- **Ensemble Model** – Combined TFT, TCN, and ETS for better accuracy  

### 📌 **3. Model Training & Hyperparameter Tuning**  
- **Loss Function**: Mean Absolute Error (MAE)  
- **Optimizer**: Adam (with adaptive learning rate)  
- **Regularization**: Dropout & L1/L2 penalties  
- **Grid Search**: For hyperparameter tuning  

### 📌 **4. Explainability & Visualization**  
- **SHAP Analysis** – Identified critical features affecting bpm predictions  
- **Grad-CAM & Seg-Grad-CAM** – Visualized **key areas** influencing predictions  

---

## 📊 **Evaluation Metrics**  
We evaluated model performance using:  
📉 **Mean Absolute Error (MAE)**  
📉 **Mean Squared Error (MSE)**  
📉 **Root Mean Squared Error (RMSE)**  
📈 **R² Score (Coefficient of Determination)**  

📌 **Visualization:**  
🖼 **2×3 subplot layout** for:  
1️⃣ Loss curve 📉  
2️⃣ Ground truth vs. predictions 📈  
3️⃣ Evaluation metric comparisons 📊  

---

## 💡 **Challenges & Future Work**  
⚠ **Data Alignment**: Different sampling rates for bpm & force data  
⚠ **Computational Constraints**: Running models on an **Intel i5 10th Gen laptop (No GPU)**  
⚠ **Anomaly Detection**: Refining thresholds for abnormal bpm classification  
⚠ **Real-Time Processing**: Enhancing efficiency for deployment  

📌 **Next Steps**:  
✅ Expand dataset for generalization  
✅ Implement real-time bpm estimation  
✅ Improve anomaly detection accuracy  

---

## 📜 **Conclusion**  
This project demonstrated a **deep learning-based, video-driven approach** for estimating bpm and detecting **anomalies in respiratory mechanics**. The **TFT + TCN + ETS ensemble model** provided accurate bpm estimations, and **SHAP analysis** improved interpretability.  

📌 **Future goal**: Deploy this model in **real-time health monitoring systems**.  

---

## 🛠 **Installation & Usage**  

### 📦 **Requirements**  
```bash
Python >= 3.8  
PyTorch  
OpenCV  
Optical Flow (Farneback, Lucas-Kanade, RAFT)  
SHAP  
Matplotlib & Seaborn  

