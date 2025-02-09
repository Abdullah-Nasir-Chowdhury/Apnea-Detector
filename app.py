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

# Define the TCN model
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(TCN, self).__init__()
        
        # List to hold convolutional layers
        self.convs = nn.ModuleList()
        dropout = dropout if num_layers > 1 else 0  # No dropout if only one layer
        self.dropout = nn.Dropout(dropout)
        
        # Create the convolutional layers
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size  # First layer uses input_size, others use hidden_size
            out_channels = hidden_size  # All layers have the same hidden size
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1))
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, features, timesteps)
        
        # Apply each convolutional layer followed by dropout
        for conv in self.convs:
            x = torch.relu(conv(x))
            x = self.dropout(x)  # Apply dropout after each convolution
        
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)  # Output layer
        return x

# Define the Temporal Fusion Transformer (Temporal Fusion Transformer) model
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        # Encoder and Decoder LSTMs with multiple layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)     
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True) # Attention mechanism
        self.fc = nn.Linear(hidden_size, output_size) # Fully connected output layer
        self.dropout = nn.Dropout(dropout) # Dropout layer

    def forward(self, x):
        encoder_output, _ = self.encoder(x) # Encoder output
        decoder_output, _ = self.decoder(encoder_output) # Decoder output
        attention_output, _ = self.attention(decoder_output, encoder_output, encoder_output) # Attention output
        attention_output = self.dropout(attention_output) # Apply dropout
        output = self.fc(attention_output[:, -1, :]) # Take the last time step from the attention output
        return output
    

# Build the ETSformer Class: Encoder, Trend, Seasonality, Exponential Smoothing, and Output Layer
class ETSformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(ETSformer, self).__init__()

        # Encoder: LSTM with multiple layers and dropout
        self.encoder = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0  # Dropout only applies if num_layers > 1
        )

        # Trend, Seasonality, Exponential Modules
        self.trend_module = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)  # Dropout in the trend module
        )
        self.seasonality_module = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)  # Dropout in the seasonality module
        )
        self.exponential_module = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)  # Dropout in the exponential module
        )

        self.fc = nn.Linear(hidden_size, output_size) # Fully connected layer for output

    def forward(self, x):
        encoder_output, _ = self.encoder(x) # Encode the input sequence
        trend = self.trend_module(encoder_output )# Trend Component
        # Seasonality Component
        freq = fft.fft(encoder_output, dim=1)  # Frequency domain transformation
        seasonality = fft.ifft(self.seasonality_module(torch.abs(freq)), dim=1).real
        exponential = torch.sigmoid(self.exponential_module(encoder_output)) # Exponential Smoothing Component
        combined = trend + seasonality + exponential # Combine the components
        # Output layer: Use the last time step for predictions
        output = self.fc(combined[:, -1, :])
        return output

# Updated BiLSTM to handle variable layers
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0  # Dropout only applies for num_layers > 1
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 for bidirectional
        
    def forward(self, x):
        bilstm_output, _ = self.bilstm(x)
        output = self.fc(bilstm_output[:, -1, :])  # Use the last time step
        return output

class RespFusion(nn.Module):
    def __init__(self, tft_model, tcn_model, ets_model, bilstm_model, meta_learner_path=None, weights=None, strategy='stacking',):
        super(RespFusion, self).__init__()
        self.tft = tft_model
        self.tcn = tcn_model
        self.ets = ets_model
        self.bilstm = bilstm_model
        self.strategy = strategy  # 'stacking' or other strategies

        # Initialize XGBoost meta-learner
        self.meta_learner = xgb.XGBRegressor()  # Or XGBClassifier for classification

        # Load the meta-learner if a path is provided
        if meta_learner_path is not None:
            self.meta_learner.load_model(meta_learner_path)
            print(f"Meta-learner loaded from {meta_learner_path}")

        # Storage for stacking training data
        self.stacking_features = []
        self.stacking_targets = []
        
        # Set model weights for ensembling, default to equal weights for weighted_average strategy
        if strategy == 'weighted_average':
            if weights is None:
                self.weights = [1.0, 1.0, 1.0, 1.0]
            else:
                assert len(weights) == 4, "Weights must match the number of models."
                self.weights = weights
                
        
    def forward(self, x):
        # Get predictions from each base model
        tft_output = self.tft(x).detach().cpu().numpy()
        tcn_output = self.tcn(x).detach().cpu().numpy()
        ets_output = self.ets(x).detach().cpu().numpy()
        bilstm_output = self.bilstm(x).detach().cpu().numpy()

        if self.strategy == 'stacking':
            # Combine outputs into features for the meta-learner
            features = np.column_stack((tft_output, tcn_output, ets_output, bilstm_output))
            # During inference, use the meta-learner to make predictions
            ensemble_output = self.meta_learner.predict(features)
            return torch.tensor(ensemble_output).to(x.device).float()
        
        elif self.strategy == 'voting':
            # For soft voting, calculate the average
            ensemble_output = torch.mean(torch.stack([tft_output, tcn_output, ets_output, bilstm_output], dim=0), dim=0)
            return ensemble_output

        elif self.strategy == 'weighted_average':
            # Weighted average of outputs
            ensemble_output = (
                self.weights[0] * tft_output +
                self.weights[1] * tcn_output +
                self.weights[2] * ets_output +
                self.weights[3] * bilstm_output
            ) / sum(self.weights)
            return ensemble_output
        
        elif self.strategy == 'simple_average':
            # Simple average of outputs
            ensemble_output = (tft_output + tcn_output + ets_output + bilstm_output) / 4
            return ensemble_output
        
        
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Currently supports only 'stacking', 'voting', 'weighted_average', and 'simple_average'.")

    def collect_stacking_data(self, x, y):
        """Collect base model outputs and corresponding targets for meta-learner training."""
        tft_output = self.tft(x).detach().cpu().numpy()
        tcn_output = self.tcn(x).detach().cpu().numpy()
        ets_output = self.ets(x).detach().cpu().numpy()
        bilstm_output = self.bilstm(x).detach().cpu().numpy()

        # Stack features and store
        features = np.column_stack((tft_output, tcn_output, ets_output, bilstm_output))
        self.stacking_features.append(features)
        self.stacking_targets.append(y.detach().cpu().numpy())

    def train_meta_learner(self, save_path=None):
        """Train the XGBoost meta-learner on collected data and save the model."""
        # Concatenate all collected features and targets
        X = np.vstack(self.stacking_features)
        y = np.concatenate(self.stacking_targets)

        # Train the XGBoost model
        self.meta_learner.fit(X, y)
        print("Meta-learner trained successfully!")

        # Save the trained meta-learner
        if save_path:
            self.meta_learner.save_model(save_path)
            print(f"Meta-learner saved to {save_path}")


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

    # Load state dicts for each model
    tft_model.load_state_dict(torch.load("F:/breathing rate estimation/Feb3/models/weights/tft_loss_optimized.pth"))
    print("TFT Model Loaded. All keys matched successfully.")
    tcn_model.load_state_dict(torch.load("F:/breathing rate estimation/Feb3/models/weights/tcn_loss_optimized.pth"))
    print("TCN Model Loaded. All keys matched successfully.")
    ets_model.load_state_dict(torch.load("F:/breathing rate estimation/Feb3/models/weights/etsformer_loss_optimized.pth"))
    print("ETSformer Model Loaded. All keys matched successfully.")
    bilstm_model.load_state_dict(torch.load("F:/breathing rate estimation/Feb3/models/weights/bilstm_loss_optimized.pth"))
    print("BiLSTM Model Loaded. All keys matched successfully.")
    
    tft_model.eval()
    tcn_model.eval()
    ets_model.eval()
    bilstm_model.eval()

    model = RespFusion(tft_model, tcn_model, ets_model, bilstm_model, strategy='stacking', meta_learner_path="F:/breathing rate estimation/Feb3/models/weights/respfusion_2_xgboost_meta_learner.json")
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

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video for Analysis"),
    outputs=gr.JSON(),
    title="Apnea Detection System",
    description="Upload a video to analyze breathing rate and detect conditions such as Apnea, Tachypnea, and Bradypnea."
)

demo.launch()