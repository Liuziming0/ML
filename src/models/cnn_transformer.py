"""CNN + Transformer：局部特征 + 长程依赖。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer_model import PositionalEncoding


class CNNTransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len + 16, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B,T,F) -> (B,F,T)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)
