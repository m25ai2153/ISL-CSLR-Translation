import torch
import torch.nn as nn

class LightweightCSLR(nn.Module):
    """
    1D CNN + BiLSTM model for Continuous Sign Language Recognition (CSLR).
    Uses CTC loss for unaligned sequence training.
    """
    def __init__(self, input_dim=225, num_classes=100, hidden_dim=128, num_layers=2):
        super(LightweightCSLR, self).__init__()
        
        # Temporal feature extraction via 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        
        # Sequential temporal modeling via Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Classification projection layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x_conv = x.transpose(1, 2)  # Reshape for Conv1D: (batch_size, input_dim, seq_len)
        x_conv = self.relu(self.conv1d(x_conv)).transpose(1, 2)  # Back to (batch_size, seq_len, hidden_dim)
        
        lstm_out, _ = self.bilstm(x_conv)  # (batch_size, seq_len, hidden_dim * 2)
        logits = self.fc(lstm_out)          # (batch_size, seq_len, num_classes)
        return logits