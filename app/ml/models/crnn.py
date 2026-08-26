from typing import Optional
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for text line recognition.
    Architecture:
        Input: [B, 1, H=32, W]
        CNN: Feature extraction & height reduction: [B, 1, 32, W] -> [B, 512, 1, W/4]
        Dimension Collapse: Squeeze height dimension -> [B, W/4, 512]
        BiLSTM: 2-layer Bidirectional LSTM -> [B, W/4, 512]
        Linear: Linear projection -> [B, W/4, num_classes] (or [W/4, B, num_classes] for CTC)
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

        # 1. CNN Feature Extractor
        # Target input height is 32.
        # Layer 1: [B, 1, 32, W] -> [B, 64, 16, W/2]
        # Layer 2: [B, 64, 16, W/2] -> [B, 128, 8, W/4]
        # Layer 3: [B, 128, 8, W/4] -> [B, 256, 8, W/4]
        # Layer 4: [B, 256, 8, W/4] -> [B, 256, 4, W/4] (pool kernel (2, 1))
        # Layer 5: [B, 256, 4, W/4] -> [B, 512, 4, W/4]
        # Layer 6: [B, 512, 4, W/4] -> [B, 512, 2, W/4] (pool kernel (2, 1))
        # Layer 7: [B, 512, 2, W/4] -> [B, 512, 1, W/4] (conv kernel (2, 1))
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
            # Conv 7 (height-collapsing convolution)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # -> [B, 512, 1, W/4]
        )

        # 2. Sequence Recurrent Network (BiLSTM)
        # Input features to LSTM = 512 (CNN output channels)
        # Output features = lstm_hidden * 2 = 512 (bidirectional)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Linear Classification Projection
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor, time_major: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape [B, 1, H=32, W].
            time_major: If True, returns logits shaped [T, B, num_classes] for PyTorch CTCLoss.
                        If False, returns logits shaped [B, T, num_classes].

        Returns:
            Logits tensor of shape [T, B, C] or [B, T, C].
        """
        # 1. Extract CNN features -> [B, 512, 1, W_seq]
        features = self.cnn(x)

        # 2. Collapse height dimension (H should be 1)
        # Squeeze H=1 -> [B, 512, W_seq] -> Transpose to [B, W_seq, 512]
        b, c, h, w_seq = features.size()
        assert h == 1, f"Expected height to collapse to 1, got {h}. Input height must be 32."
        features = features.squeeze(2).permute(0, 2, 1)  # [B, W_seq, 512]

        # 3. BiLSTM Sequence Modeling -> [B, W_seq, 2 * lstm_hidden]
        recurrent_out, _ = self.lstm(features)

        # 4. Linear Projection -> [B, W_seq, num_classes]
        logits = self.fc(recurrent_out)

        if time_major:
            # Transpose to [W_seq, B, num_classes]
            return logits.permute(1, 0, 2)
        return logits
