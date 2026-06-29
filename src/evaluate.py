"""测试集评估与预测曲线绘制。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import Horizon, build_dataset, inverse_transform_target
from src.metrics import mae, mse
from src.train import get_device, load_checkpoint
from src.utils import load_config, resolve_path

HORIZON_CN = {"short": "短期", "long": "长期"}


def setup_matplotlib_cn() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def predict_loader(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"]
            pred = model(x).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def evaluate_checkpoint(
    ckpt_path: Path,
    horizon: Horizon,
    config: dict[str, Any] | None = None,
    plot_path: Path | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    model, ckpt = load_checkpoint(ckpt_path, cfg)

    test_ds = build_dataset("test", horizon, cfg)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    pred_norm, true_norm = predict_loader(model, loader)
    pred = inverse_transform_target(pred_norm, cfg)
    true = inverse_transform_target(true_norm, cfg)

    result = {
        "model": ckpt["model_name"],
        "horizon": horizon.value,
        "seed": ckpt["seed"],
        "mse": mse(true, pred),
        "mae": mae(true, pred),
        "n_test_samples": len(test_ds),
    }

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_prediction(true, pred, result, plot_path)

    return result


def _plot_prediction(
    true: np.ndarray,
    pred: np.ndarray,
    meta: dict[str, Any],
    save_path: Path,
) -> None:
    setup_matplotlib_cn()
    idx = len(true) // 2
    y_true = true[idx]
    y_pred = pred[idx]
    days = np.arange(1, len(y_true) + 1)
    horizon_cn = HORIZON_CN.get(meta["horizon"], meta["horizon"])

    plt.figure(figsize=(10, 4))
    plt.plot(days, y_true, label="真实值", linewidth=2)
    plt.plot(days, y_pred, label="预测值", linewidth=2, linestyle="--")
    plt.xlabel("天数")
    plt.ylabel("全局有功功率 (kW·天)")
    plt.title(
        f"{meta['model']} | {horizon_cn} | seed={meta['seed']} | "
        f"MSE={meta['mse']:.2f} MAE={meta['mae']:.2f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(
    results_dir: Path,
    horizon: Horizon,
    save_path: Path,
) -> None:
    """叠加各模型在同一样本上的预测曲线。"""
    setup_matplotlib_cn()
    models = ["lstm", "transformer", "cnn_transformer"]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        run_dir = results_dir / f"{model_name}_{horizon.value}"
        plots = sorted(run_dir.glob("plot_seed*.png"))
        if not plots:
            ax.set_title(f"{model_name}：未找到曲线图")
            continue
        img = plt.imread(plots[0])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(model_name)

    fig.suptitle(f"模型对比（{HORIZON_CN.get(horizon.value, horizon.value)}）")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
