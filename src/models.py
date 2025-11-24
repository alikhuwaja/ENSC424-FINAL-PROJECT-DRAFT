import torch
import torch.nn as nn
from . import config

class CRNN(nn.Module):
    """
    Baseline CRNN for Speech Emotion Recognition.
    Input: (B, 1, NUM_MEL, T)
    Output: (B, NUM_CLASSES)
    """

    def __init__(self, num_mels=config.NUM_MEL, num_classes=config.NUM_CLASSES):
        super().__init__()

        # ---- CNN Feature Extractor ----
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # After pooling: NUM_MEL // 4 frequency bins
        cnn_output_freq = num_mels // 4
        cnn_output_channels = 64

        self.gru = nn.GRU(
            input_size=cnn_output_channels * cnn_output_freq,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, num_mels, T)
        z = self.cnn(x)       # (B, C, F, T')
        B, C, F, T = z.shape

        z = z.permute(0, 3, 1, 2)  # (B, T', C, F)
        z = z.reshape(B, T, C * F)  # (B, T', feat)

        out, _ = self.gru(z)
        out = out[:, -1, :]  # last time step
        return self.fc(out)

class SERTransformer(nn.Module):
    """
    Lightweight Transformer Encoder model for Speech Emotion Recognition.
    Input shape: (B, 1, NUM_MEL, T)
    """

    def __init__(self,
                 num_mels=config.NUM_MEL,
                 num_classes=config.NUM_CLASSES,
                 embed_dim=128,
                 num_heads=4,
                 ff_dim=256,
                 num_layers=3):
        super().__init__()

        # ---- CNN front-end (optional, small) ----
        # Helps reduce time dimension & learn local patterns before Transformer
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # halve time axis
        )

        transformer_input_dim = 32 * num_mels

        self.proj = nn.Linear(transformer_input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,  # IMPORTANT for (B, T, D)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ff_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, mel, T)
        z = self.conv(x)      # (B, 32, mel, T//2)
        B, C, F, T = z.shape

        z = z.permute(0, 3, 1, 2)        # (B, T, C, F)
        z = z.reshape(B, T, C * F)       # (B, T, transformer_input)
        z = self.proj(z)                 # (B, T, embed_dim)

        out = self.transformer(z)        # (B, T, embed_dim)
        out = out.mean(dim=1)            # average pooling (global)

        return self.fc(out)