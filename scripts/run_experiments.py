"""批量实验：3 模型 × 2 预测长度 × N 个 seed。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import Horizon
from src.evaluate import evaluate_checkpoint, setup_matplotlib_cn
from src.metrics import summarize_runs
from src.train import train_one_model
from src.utils import load_config, resolve_path, save_json

MODELS = ["lstm", "transformer", "cnn_transformer"]
HORIZON_CN = {"short": "短期", "long": "长期"}


def run_experiments(
    config_path: str | None = None,
    quick: bool = False,
    models: list[str] | None = None,
) -> dict:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    seeds = [train_cfg["seeds"][0]] if quick else train_cfg["seeds"]
    if quick:
        cfg = {
            **cfg,
            "train": {**train_cfg, "epochs": 3, "patience": 2},
        }
        train_cfg = cfg["train"]

    results_root = resolve_path(cfg["paths"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)

    model_list = models or MODELS
    all_runs: list[dict] = []
    summary: dict = {}

    for model_name in model_list:
        for horizon in (Horizon.SHORT, Horizon.LONG):
            key = f"{model_name}_{horizon.value}"
            summary[key] = {"mse": [], "mae": [], "runs": []}
            horizon_cn = HORIZON_CN[horizon.value]

            for seed in seeds:
                run_dir = results_root / key / f"seed_{seed}"
                print(f"\n=== {model_name} | {horizon_cn} | seed={seed} ===")

                train_out = train_one_model(
                    model_name=model_name,
                    horizon=horizon,
                    seed=seed,
                    config=cfg,
                    save_dir=run_dir,
                )
                ckpt_path = Path(train_out["meta"]["checkpoint"])
                plot_path = run_dir / f"plot_seed{seed}.png"
                eval_result = evaluate_checkpoint(ckpt_path, horizon, cfg, plot_path)

                run_record = {
                    **train_out["meta"],
                    "test_mse": eval_result["mse"],
                    "test_mae": eval_result["mae"],
                    "plot": str(plot_path),
                }
                all_runs.append(run_record)
                summary[key]["mse"].append(eval_result["mse"])
                summary[key]["mae"].append(eval_result["mae"])
                summary[key]["runs"].append(run_record)

                print(f"  测试 MSE={eval_result['mse']:.4f}  MAE={eval_result['mae']:.4f}")

            summary[key]["mse_stats"] = summarize_runs(summary[key]["mse"])
            summary[key]["mae_stats"] = summarize_runs(summary[key]["mae"])
            print(
                f"  >> {key} MSE: {summary[key]['mse_stats']['mean']:.4f} "
                f"± {summary[key]['mse_stats']['std']:.4f}"
            )
            print(
                f"  >> {key} MAE: {summary[key]['mae_stats']['mean']:.4f} "
                f"± {summary[key]['mae_stats']['std']:.4f}"
            )

    out = {"runs": all_runs, "summary": summary}
    save_json(out, results_root / "summary.json")
    _save_summary_table(summary, results_root / "summary_table.txt")
    for horizon in (Horizon.SHORT, Horizon.LONG):
        _plot_horizon_comparison(results_root, horizon, model_list, seeds[0])
    print(f"\n结果已保存至 {results_root}")
    return out


def _plot_horizon_comparison(
    results_root: Path,
    horizon: Horizon,
    models: list[str],
    seed: int,
) -> None:
    """同一测试样本上对比各模型预测曲线。"""
    import matplotlib.pyplot as plt

    setup_matplotlib_cn()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import build_dataset, inverse_transform_target
    from src.train import load_checkpoint

    cfg = load_config()
    test_ds = build_dataset("test", horizon, cfg)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    sample_idx = len(test_ds) // 2
    batch_idx = sample_idx // 32
    in_batch_idx = sample_idx % 32
    horizon_cn = HORIZON_CN[horizon.value]

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.2 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        ckpt = results_root / f"{model_name}_{horizon.value}" / f"seed_{seed}" / "model.pt"
        if not ckpt.exists():
            ax.set_title(f"{model_name}：缺少 checkpoint")
            continue
        model, meta = load_checkpoint(ckpt, cfg)
        device = next(model.parameters()).device
        preds, trues = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                y = batch["y"]
                pred = model(x).cpu()
                preds.append(pred)
                trues.append(y)
                if i == batch_idx:
                    break
        pred_norm = preds[batch_idx][in_batch_idx].numpy()
        true_norm = trues[batch_idx][in_batch_idx].numpy()
        pred = inverse_transform_target(pred_norm, cfg)
        true = inverse_transform_target(true_norm, cfg)
        days = np.arange(1, len(true) + 1)
        ax.plot(days, true, label="真实值", linewidth=2)
        ax.plot(days, pred, label="预测值", linewidth=2, linestyle="--")
        ax.set_ylabel("功率 (kW·天)")
        ax.set_title(model_name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("预测天数")
    fig.suptitle(f"模型对比 — {horizon_cn}（seed={seed}）")
    plt.tight_layout()
    out = results_root / f"comparison_{horizon.value}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已保存对比图：{out}")


def _save_summary_table(summary: dict, path: Path) -> None:
    lines = [
        f"{'模型':<18} {'任务':<8} {'MSE mean±std':<22} {'MAE mean±std':<22}",
        "-" * 72,
    ]
    for key, stats in summary.items():
        model, horizon = key.rsplit("_", 1)
        mse_s = stats["mse_stats"]
        mae_s = stats["mae_stats"]
        horizon_cn = HORIZON_CN.get(horizon, horizon)
        lines.append(
            f"{model:<18} {horizon_cn:<8} "
            f"{mse_s['mean']:.2f}±{mse_s['std']:.2f}{'':>8} "
            f"{mae_s['mean']:.2f}±{mae_s['std']:.2f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行预测实验")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--quick", action="store_true", help="快速测试：1 个 seed、3 个 epoch")
    parser.add_argument("--models", nargs="+", default=None, choices=MODELS, help="指定模型")
    args = parser.parse_args()
    run_experiments(args.config, quick=args.quick, models=args.models)


if __name__ == "__main__":
    main()
