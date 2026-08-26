from typing import Optional
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN architecture for OCR text-line recognition training.
    CNN feature extractor -> Squeeze height -> BiLSTM -> Linear projection.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # CNN Feature Extractor (Target input height: 32)
        self.cnn = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 16, W/2]
            # Conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 128, 8, W/4]
            # Conv 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Conv 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 256, 4, W/4]
            # Conv 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Conv 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 512, 2, W/4]
            # Conv 7 (height reduction)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # -> [B, 512, 1, W/4]
        )

        # BiLSTM sequence layer
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Linear Projection to vocabulary logits
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor, time_major: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape [B, 1, H=32, W].
            time_major: If True, returns [T, B, num_classes] for PyTorch CTCLoss.
        """
        features = self.cnn(x)
        b, c, h, w_seq = features.size()
        assert h == 1, f"Expected height to collapse to 1, got {h}"
        features = features.squeeze(2).permute(0, 2, 1)  # [B, W_seq, 512]

        recurrent_out, _ = self.lstm(features)  # [B, W_seq, 512]
        logits = self.fc(recurrent_out)  # [B, W_seq, num_classes]

        if time_major:
            return logits.permute(1, 0, 2)  # [W_seq, B, num_classes]
        return logits
