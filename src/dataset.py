"""滑动窗口 Dataset：90→90 / 90→365 多步预测。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import load_config, load_scaler, resolve_path


class Horizon(str, Enum):
    SHORT = "short"  # 90→90
    LONG = "long"    # 90→365


def get_output_len(horizon: Horizon, config: dict | None = None) -> int:
    cfg = config or load_config()
    w = cfg["window"]
    return w["short_output_len"] if horizon == Horizon.SHORT else w["long_output_len"]


def get_window_params(config: dict | None = None) -> tuple[int, int, int]:
    cfg = config or load_config()
    w = cfg["window"]
    return w["input_len"], w["short_output_len"], w["stride"]


class PowerForecastDataset(Dataset):
    """滑动窗口：X (input_len, n_features)，y (output_len,)。"""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        input_len: int = 90,
        output_len: int = 90,
        stride: int = 1,
        feature_scaler: Any | None = None,
        target_scaler: Any | None = None,
        fit_scaler: bool = False,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride

        target_idx = feature_cols.index(target_col)
        features = df[feature_cols].to_numpy(dtype=np.float32)
        self.dates = df["date"].values

        if fit_scaler:
            from sklearn.preprocessing import StandardScaler

            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(features)
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(features[:, [target_idx]])
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features).astype(np.float32)

        self.features = features
        self.target_idx = target_idx

        total = input_len + output_len
        self.indices: list[int] = list(range(0, len(df) - total + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        mid = start + self.input_len
        end = mid + self.output_len

        x = self.features[start:mid]
        y = self.features[mid:end, self.target_idx]

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "input_start_date": str(pd.Timestamp(self.dates[start]).date()),
            "output_start_date": str(pd.Timestamp(self.dates[mid]).date()),
        }


def load_split_csv(split: str, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path_key = "train_csv" if split == "train" else "test_csv"
    path = resolve_path(cfg["paths"][path_key])
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 python scripts/run_preprocess.py")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_dataset(
    split: str,
    horizon: Horizon,
    config: dict | None = None,
    stride: int | None = None,
) -> PowerForecastDataset:
    cfg = config or load_config()
    df = load_split_csv(split, cfg)

    scaler_bundle = load_scaler(resolve_path(cfg["paths"]["scaler_path"]))
    feature_cols = scaler_bundle["feature_cols"]
    target_col = scaler_bundle["target_col"]

    input_len = cfg["window"]["input_len"]
    output_len = get_output_len(horizon, cfg)
    step = stride if stride is not None else cfg["window"]["stride"]

    return PowerForecastDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        input_len=input_len,
        output_len=output_len,
        stride=step,
        feature_scaler=scaler_bundle["feature_scaler"],
        target_scaler=scaler_bundle["target_scaler"],
        fit_scaler=False,
    )


def build_dataloader(
    split: str,
    horizon: Horizon,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    config: dict | None = None,
    stride: int | None = None,
) -> DataLoader:
    dataset = build_dataset(split, horizon, config, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def inverse_transform_target(
    values: np.ndarray | torch.Tensor,
    config: dict | None = None,
) -> np.ndarray:
    """目标反标准化，还原为 kW 尺度。"""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    cfg = config or load_config()
    scaler_bundle = load_scaler(resolve_path(cfg["paths"]["scaler_path"]))
    target_scaler = scaler_bundle["target_scaler"]
    flat = values.reshape(-1, 1)
    return target_scaler.inverse_transform(flat).reshape(values.shape)


def describe_dataset(ds: PowerForecastDataset) -> dict[str, Any]:
    sample = ds[0]
    return {
        "样本数": len(ds),
        "输入长度": ds.input_len,
        "输出长度": ds.output_len,
        "特征数": sample["x"].shape[1],
        "X 形状": list(sample["x"].shape),
        "Y 形状": list(sample["y"].shape),
        "首个输入起始日": sample["input_start_date"],
        "首个输出起始日": sample["output_start_date"],
    }
