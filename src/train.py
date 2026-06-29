"""训练循环：按时间顺序划分 train/val。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import Horizon, build_dataset
from src.models.factory import build_model
from src.utils import load_config, set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chronological_split(dataset_len: int, val_ratio: float) -> tuple[list[int], list[int]]:
    val_size = max(1, int(dataset_len * val_ratio))
    train_size = dataset_len - val_size
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, dataset_len))
    return train_idx, val_idx


def train_one_model(
    model_name: str,
    horizon: Horizon,
    seed: int,
    config: dict[str, Any] | None = None,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    train_cfg = cfg["train"]
    set_seed(seed)

    device = get_device()
    input_len = cfg["window"]["input_len"]
    output_len = (
        cfg["window"]["short_output_len"]
        if horizon == Horizon.SHORT
        else cfg["window"]["long_output_len"]
    )

    full_ds = build_dataset("train", horizon, cfg)
    n_features = full_ds[0]["x"].shape[1]
    train_idx, val_idx = _chronological_split(len(full_ds), train_cfg["val_ratio"])

    batch_size = train_cfg["batch_size_long"] if horizon == Horizon.LONG else train_cfg["batch_size"]
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(model_name, n_features, input_len, output_len, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    history: list[dict[str, float]] = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result: dict[str, Any] = {
        "model": model_name,
        "horizon": horizon.value,
        "seed": seed,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "history": history,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "horizon": horizon.value,
                "seed": seed,
                "n_features": n_features,
                "input_len": input_len,
                "output_len": output_len,
                "best_val_loss": best_val,
            },
            ckpt_path,
        )
        result["checkpoint"] = str(ckpt_path)

    return {"model": model, "meta": result}


def load_checkpoint(path: Path, config: dict[str, Any] | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = config or load_config()
    ckpt = torch.load(path, map_location=get_device(), weights_only=False)
    model = build_model(
        ckpt["model_name"],
        ckpt["n_features"],
        ckpt["input_len"],
        ckpt["output_len"],
        cfg,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(get_device())
    model.eval()
    return model, ckpt
