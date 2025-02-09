import torch
import torch.nn as nn

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