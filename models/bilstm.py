import torch.nn as nn


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