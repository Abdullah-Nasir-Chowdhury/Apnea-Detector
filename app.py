import gradio as gr
import torch
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from collections import Counter
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
import torch.fft as fft
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
import time
from models.temporal_fusion_transformer import TemporalFusionTransformer
from models.tcn import TCN
from models.etsformer import ETSformer
from models.bilstm import BiLSTM
from models.respfusion import RespFusion 

def process_video(video_path):
    # Parameters
    roi_coordinates = None  # Manual ROI selection
    fps = 30  # Frames per second
    feature_window = 10  # Window for feature aggregation (seconds)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error opening video file"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select ROI
    ret, first_frame = cap.read()
    if not ret:
        return "Failed to read first frame"

    if roi_coordinates is None:
        roi_coordinates = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi_coordinates)

    prev_gray = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    motion_magnitude, direction_angles = [], []

    frame_skip = 2
    scale_factor = 0.5

    roi = cv2.resize(first_frame[y:y+h, x:x+w], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index % frame_skip != 0:
            continue

        roi = cv2.resize(frame[y:y+h, x:x+w], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 2, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motion_magnitude.append(np.mean(magnitude))
        direction_angles.extend(angle.flatten())

        prev_gray = gray

    cap.release()

    window_length = min(len(motion_magnitude) - 1, 31)
    if window_length % 2 == 0:
        window_length += 1
    smoothed_magnitude = savgol_filter(motion_magnitude, window_length=window_length, polyorder=3)

    features = []
    window_size = feature_window * (fps // frame_skip)
    num_windows = len(smoothed_magnitude) // window_size

    for i in range(num_windows):
        start, end = i * window_size, (i + 1) * window_size
        window_magnitude = smoothed_magnitude[start:end]
        peaks, _ = find_peaks(window_magnitude)
        troughs, _ = find_peaks(-window_magnitude)

        features.append([
            np.mean(window_magnitude), np.std(window_magnitude), np.max(window_magnitude), np.min(window_magnitude),
            len(peaks), np.mean(window_magnitude[peaks]) if len(peaks) > 0 else 0,
            np.mean(np.diff(peaks)) / (fps // frame_skip) if len(peaks) > 1 else 0,
            (np.mean(window_magnitude[peaks]) - np.mean(window_magnitude[troughs])) if peaks.size > 0 and troughs.size > 0 else 0,
            Counter((np.array(direction_angles[start:end]) / (np.pi / 4)).astype(int) % 8).most_common(1)[0][0] if len(direction_angles) > 0 else -1,
            np.argmax(np.abs(np.fft.fft(window_magnitude))[1:len(window_magnitude)//2]) / window_size
        ])

    features_df = pd.DataFrame(features, columns=[
        "mean_magnitude", "std_magnitude", "max_magnitude", "min_magnitude", "peak_count", "avg_peak_height", "peak_to_peak_interval", "amplitude", "dominant_direction", "dominant_frequency"
    ])
    print("Features Extracted. Matching Model Keys.")

    X_inference = np.array([features_df.iloc[i-5:i, :].values for i in range(5, len(features_df))])
    X_inference_tensor = torch.tensor(X_inference, dtype=torch.float32)
    

    input_size, hidden_size, output_size = X_inference_tensor.shape[2], 64, 1
    tft_model = TemporalFusionTransformer(input_size, hidden_size, output_size)
    tcn_model = TCN(input_size, hidden_size, output_size)
    ets_model = ETSformer(input_size, hidden_size, output_size)
    bilstm_model = BiLSTM(input_size, hidden_size, output_size)

    weights_path = './models/weights/'
    # Load state dicts for each model
    tft_model.load_state_dict(torch.load(weights_path + "tft_loss_optimized.pth", map_location=torch.device("cpu")))
    print("TFT Model Loaded. All keys matched successfully.")
    tcn_model.load_state_dict(torch.load(weights_path + "tcn_loss_optimized.pth", map_location=torch.device("cpu")))
    print("TCN Model Loaded. All keys matched successfully.")
    ets_model.load_state_dict(torch.load(weights_path + "etsformer_loss_optimized.pth", map_location=torch.device("cpu")))
    print("ETSformer Model Loaded. All keys matched successfully.")
    bilstm_model.load_state_dict(torch.load(weights_path + "bilstm_loss_optimized.pth", map_location=torch.device("cpu")))
    print("BiLSTM Model Loaded. All keys matched successfully.")
    
    
    tft_model.eval()
    tcn_model.eval()
    ets_model.eval()
    bilstm_model.eval()

    model = RespFusion(
    tft_model, tcn_model, ets_model, bilstm_model,
    strategy='stacking',
    meta_learner_path=weights_path + "respfusion_2_xgboost_meta_learner.json"
    )
    print("Respfusion Model Loaded. Detecting Apnea.")
    with torch.no_grad():
        predictions = model(X_inference_tensor).squeeze().tolist()

    def categorize_breathing_rate(pred):
        if pred < 0.5:
            return "Apnea"
        elif pred > 20:
            return "Tachypnea"
        elif pred < 10:
            return "Bradypnea"
        else:
            return "Normal"

    categories = [categorize_breathing_rate(pred) for pred in predictions]
    overall_category = categorize_breathing_rate(sum(predictions) / len(predictions))

    return {"Breathing State per 10s": categories, "Average Breathing Rate": sum(predictions)/len(predictions), "Overall Breathing State": overall_category,}

app = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video for Analysis"),
    outputs=gr.JSON(),
    title="Apnea Detection System",
    description="Upload a video to analyze breathing rate and detect conditions such as Apnea, Tachypnea, and Bradypnea."
)

app.launch()