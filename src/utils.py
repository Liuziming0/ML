"""路径、配置、标准化与随机种子。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fit_scaler(
    train_features: np.ndarray,
    target_col_index: int = 0,
) -> tuple[StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_features)
    target_scaler = StandardScaler()
    target_scaler.fit(train_features[:, [target_col_index]])
    return feature_scaler, target_scaler


def save_scaler(
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    path: Path,
    feature_cols: list[str],
    target_col: str,
) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        path,
    )


def load_scaler(path: Path) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
