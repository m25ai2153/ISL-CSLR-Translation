import torch
import torch.nn as nn

class LightweightCSLR(nn.Module):
    def __init__(self, input_dim=288, num_classes=100, hidden_dim=128, num_layers=2):
        super(LightweightCSLR, self).__init__()
        
        # 1D CNN Feature Extractor along sequence timeline
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Bidirectional LSTM for sequence alignment
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Final classification head mapping features to CTC vocabulary
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, input_dim]
        x = x.transpose(1, 2)  # Reshape for 1D Conv: [batch_size, input_dim, sequence_length]
        x = self.conv_block(x)
        x = x.transpose(1, 2)  # Reshape for LSTM: [batch_size, reduced_sequence_length, 256]
        
        lstm_out, _ = self.bilstm(x)
        logits = self.classifier(lstm_out)  # [batch_size, reduced_sequence_length, num_classes]
        
        # Output log-probabilities for CTC loss
        return torch.log_softmax(logits, dim=-1)