import torch
import torch.nn as nn
import torch.fft as fft

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

