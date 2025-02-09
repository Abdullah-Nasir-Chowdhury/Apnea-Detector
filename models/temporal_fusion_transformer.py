import torch.nn as nn


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