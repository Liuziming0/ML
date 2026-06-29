"""模型工厂。"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.cnn_transformer import CNNTransformerForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.transformer_model import TransformerForecaster

MODEL_REGISTRY = {
    "lstm": LSTMForecaster,
    "transformer": TransformerForecaster,
    "cnn_transformer": CNNTransformerForecaster,
}


def build_model(
    name: str,
    n_features: int,
    input_len: int,
    output_len: int,
    cfg: dict[str, Any],
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}，可选：{list(MODEL_REGISTRY)}")
    model_cfg = cfg.get("train", {}).get("models", {}).get(name, {})
    cls = MODEL_REGISTRY[name]
    return cls(
        n_features=n_features,
        input_len=input_len,
        output_len=output_len,
        **model_cfg,
    )
